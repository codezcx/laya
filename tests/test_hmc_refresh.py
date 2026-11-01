import hashlib
import json
import os

import pytest

from scripts import prepare_multitask_splits as splits


@pytest.fixture
def official_hmc(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, 'MNE_ROOT', tmp_path)
    root = tmp_path / 'HMC'
    recordings = root / 'recordings'
    recordings.mkdir(parents=True)
    names = []
    # Tiny synthetic files exercise integrity gates, not physiological processing.
    for subject in range(1, 152):
        for suffix in ('.edf', '_sleepscoring.edf', '_sleepscoring.txt'):
            path = recordings / f'SN{subject:03d}{suffix}'
            path.write_bytes(f'synthetic {subject} {suffix}'.encode())
            names.append(str(path.relative_to(root)))
    (root / 'RECORDS').write_text(''.join(f'recordings/SN{s:03d}.edf\n' for s in range(1, 152)))
    names.append('RECORDS')
    for name in ('LICENSE.txt', 'Subjects_info_aggregated.txt', 'HMCdatabase_quickcheck.xml'):
        (root / name).write_text('synthetic metadata')
        names.append(name)
    entries = {name: splits.sha(root / name) for name in names}
    (root / 'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name, digest in entries.items()))
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'official_manifest.json').write_text(json.dumps(dict(
        url='https://physionet.org/files/hmc-sleep-staging/1.1/SHA256SUMS.txt',
        manifest_sha256=splits.sha(root / 'SHA256SUMS.txt'), files=entries)))
    report = run / 'complete.json'
    report.write_text(json.dumps(dict(status='complete', recordings=151,
        annotation_txt=151, annotation_edf=151, official_files=457,
        final_verified_files=457, final_mismatches=[], original_files_changed=[],
        dataset_root=str(root))))
    return root, report


def test_complete_release_is_rehashed(official_hmc):
    _, report = official_hmc
    result = splits.verify_complete_hmc(report)
    assert result['verified_recordings'] == 151
    assert result['verified_files'] == 457


def test_stale_complete_report_does_not_hide_changed_file(official_hmc):
    root, report = official_hmc
    (root / 'recordings/SN001.edf').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        splits.verify_complete_hmc(report)


def test_extra_unofficial_signal_is_rejected(official_hmc):
    root, report = official_hmc
    (root / 'recordings/SN999.edf').write_bytes(b'extra')
    with pytest.raises(ValueError, match='extra records'):
        splits.verify_complete_hmc(report)


def test_partial_download_is_rejected_before_allocation(official_hmc):
    _, report = official_hmc
    state = json.loads(report.read_text())
    state['status'] = 'downloading'
    state['recordings'] = 109
    report.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='complete 151-record'):
        splits.verify_complete_hmc(report)


def test_official_hmc_split_has_103_24_24_disjoint_subjects():
    rows = [dict(subject=s, record_id=f'SN{s:03d}') for s in range(1, 152)]
    first = splits.allocate('hmc', rows)
    second = splits.allocate('hmc', rows)
    assert first.equals(second)
    assert first.groupby('split')['subject'].nunique().to_dict() == dict(train=103, valid=24, test=24)
    assert first.groupby('subject')['split'].nunique().max() == 1


def test_refresh_preserves_other_dataset_bytes(tmp_path):
    base, output = tmp_path / 'base', tmp_path / 'new'
    (base / 'other').mkdir(parents=True)
    output.mkdir()
    contents = b'record_id,subject,split\na,1,train\nb,2,test\n'
    (base / 'other/records.csv').write_bytes(contents)
    manifest = dict(records_sha256=hashlib.sha256(contents).hexdigest())
    (base / 'other/manifest.json').write_text(json.dumps(manifest))
    previous = dict(upstream_commit=splits.COMMIT, datasets={'hmc': {}, 'other': manifest},
                    tuh_overlap_audit={'risk': 'retained'},
                    blocking_decisions=['HMC missing recordings: old subset', 'TUH identity unresolved'])
    (base / 'summary.json').write_text(json.dumps(previous))
    overall = dict(datasets={'hmc': {'new': True}})
    splits.retain_other_allocations(overall, output, base, splits.sha(base / 'summary.json'))
    assert (output / 'other/records.csv').read_bytes() == contents
    assert (base / 'other/records.csv').read_bytes() == contents
    assert overall['datasets']['hmc'] == {'new': True}
    assert overall['tuh_overlap_audit'] == previous['tuh_overlap_audit']
    assert overall['blocking_decisions'] == ['TUH identity unresolved']
    assert overall['refresh']['changed_datasets'] == ['hmc']


def test_changed_base_allocation_is_rejected(tmp_path):
    base, output = tmp_path / 'base', tmp_path / 'new'
    base.mkdir()
    output.mkdir()
    (base / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='Base split summary changed'):
        splits.retain_other_allocations({'datasets': {'hmc': {}}}, output, base, '0' * 64)


@pytest.mark.parametrize('state,expected', [
    ({'status': 'incomplete', 'downloader_pid': 123}, 'terminal state'),
    ({'status': 'downloading'}, 'no verifiable process handle'),
    ({'status': 'downloading', 'downloader_pid': os.getpid()}, 'no longer matches'),
])
def test_wait_rejects_failed_or_unrelated_process(tmp_path, state, expected):
    (tmp_path / 'progress.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match=expected):
        splits.wait_for_complete_hmc(tmp_path / 'complete.json')


def test_wait_does_not_need_live_process_after_completion(official_hmc):
    _, report = official_hmc
    splits.wait_for_complete_hmc(report)
    assert splits.verify_complete_hmc(report)['verified_recordings'] == 151
