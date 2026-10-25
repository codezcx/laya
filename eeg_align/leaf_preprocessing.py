"""Matched 2a input control using LEAF's raw-signal preprocessing.

Only the existing fold's training/validation T-session trials are exported.
LEAF's subject-level robust scaling uses all unlabeled trials of each subject;
this is a batch/transductive input convention, not single-trial normalization.
"""
import json
from pathlib import Path

import numpy as np


def raw_2a_epochs(data):
    """Use LEAF's sample indexing/CAR, selecting labeled runs (also handles A04T)."""
    samples, labels, runs = [], [], []
    for run_index, entry in enumerate(data['data'][0]):
        run = entry[0, 0]
        targets = np.asarray(run[2]).reshape(-1)
        if not len(targets):
            continue
        starts = np.asarray(run[1]).reshape(-1)
        assert len(starts) == len(targets) and set(targets) <= {1, 2, 3, 4}
        raw = np.asarray(run[0])[:, :22]
        assert raw.shape[1] == 22
        for event, label in zip(starts, targets):
            # Deliberately match upstream proc_one_mat's verbatim MATLAB index.
            sample = raw[int(event):int(event) + 1500].T.copy()
            assert sample.shape == (22, 1500) and np.isfinite(sample).all()
            sample -= sample.mean(axis=0, keepdims=True)
            samples.append(sample)
            labels.append(int(label) - 1)
        runs.append(dict(run_index=run_index, trials=len(targets)))
    assert samples, 'No labeled runs in MAT file'
    return np.stack(samples), np.asarray(labels, dtype=int), runs


def prepare_leaf_2a(raw_root, out):
    import mne
    import scipy.io
    from .leaf_runtime import CONFIG, UP, digest
    from .signal_data import ROOT as SIGNAL_ROOT, load_rows
    from leaf_datasets.MI_BCIC_IV2a import CH_NAMES, proc_one_mat
    from leaf_datasets.shared import Param, pipeline

    raw_root, out = Path(raw_root), Path(out)
    rows = sorted([r for split in ['train', 'validation']
                   for r in load_rows(split, CONFIG['fold'])
                   if r['dataset'] == 'BNCI2014001'], key=lambda r: r['source_index'])
    subjects = sorted({r['subject'] for r in rows}, key=float)
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['task'] for r in rows}) == 1
    assert not ({r['subject'] for r in rows if r['split'] == 'train'} &
                {r['subject'] for r in rows if r['split'] == 'validation'})
    raw_files = {s: raw_root/f'A{int(float(s)):02d}T.mat' for s in subjects}
    assert all(p.is_file() for p in raw_files.values())
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(
        dataset='BNCI2014001', fold=CONFIG['fold'], n=len(rows), subjects=subjects,
        raw_files={s: str(p.resolve()) for s,p in raw_files.items()},
        counts={split: sum(r['split'] == split for r in rows) for split in ['train','validation']},
        raw_sessions='T only; existing benchmark trial IDs, labels and subject splits retained',
        normalization='Upstream pipeline over all 288 unlabeled trials of each subject separately; '
                      'validation-subject normalization is transductive, with no label use',
        processing='22-channel CAR; MNE 0.3-40 Hz FIR on 6-second raw epochs; MNE resample '
                   '250 to 200 Hz; keep seconds 2-6; upstream shared.pipeline clipping, '
                   'subject median/IQR and 65-channel mapping',
        differences_from_upstream='Use only T session to preserve current trial population; '
                                  'select all labeled runs instead of skipping first three '
                                  '(A04T has only one calibration run). Other sample indexing unchanged.',
        no_test_signal_access=True, mne_version=mne.__version__, upstream_commit=CONFIG['upstream_commit'],
        source_sha256={str(p): digest(p) for p in [UP/'leaf_datasets/MI_BCIC_IV2a.py',
                      UP/'leaf_datasets/shared.py', UP/'leaf_datasets/electrode_unifier.py', Path(__file__)]},
        manifests_sha256={split: digest(SIGNAL_ROOT/f"fold{CONFIG['fold']}"/f'{split}.jsonl')
                          for split in ['train','validation']})
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    target = np.lib.format.open_memmap(out/'signals.npy', mode='w+', dtype=np.float32,
                                      shape=(len(rows), 65, 800))
    records, reports, offset = [], {}, 0
    for subject in subjects:
        rr = [r for r in rows if r['subject'] == subject]
        data = scipy.io.loadmat(raw_files[subject])
        x, y, runs = raw_2a_epochs(data)
        assert len(rr) == len(y) == 288
        assert np.array_equal(y, [r['label'] for r in rr]), 'Raw/cache trial order or labels differ'
        native_x, native_y = proc_one_mat(data)
        # Confirm the implementation matches upstream wherever its run selection is valid.
        native_run_selection_matches = len(native_y) == len(y)
        if native_run_selection_matches:
            assert np.array_equal(x, native_x) and np.array_equal(y, native_y)
        else:
            assert int(float(subject)) == 4 and len(native_y) == 192
            np.testing.assert_array_equal(native_y, y[96:])
            np.testing.assert_array_equal(native_x, x[96:])
        car_error = float(np.abs(x.mean(1)).max())
        epoch = mne.EpochsArray(x, mne.create_info(CH_NAMES, sfreq=250, ch_types='eeg'), verbose=False)
        epoch.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
        epoch.resample(Param.resample, npad='auto', verbose=False)
        processed = epoch.get_data().astype(np.float32)[:, :, 400:1200]
        processed = pipeline(processed, CH_NAMES)
        assert processed.shape == (288,65,800) and np.isfinite(processed).all()
        target[offset:offset+len(rr)] = processed
        for i,r in enumerate(rr):
            records.append(dict(r, prepared_index=offset+i))
        reports[subject] = dict(n=288, split=rr[0]['split'], raw_sha256=digest(raw_files[subject]),
                                raw_labels_exactly_match_cache=True, runs=runs,
                                upstream_raw_epochs_exact_on_common_runs=True,
                                recovered_trials=0 if native_run_selection_matches else 96,
                                maximum_CAR_channel_mean_error=car_error,
                                processed_abs_max=float(np.abs(processed).max()))
        offset += len(rr)
        print(json.dumps(dict(subject=subject, prepared=offset, total=len(rows),
                              recovered_trials=reports[subject]['recovered_trials'])), flush=True)
        del data, x, native_x, epoch, processed
    assert offset == len(rows)
    target.flush()
    del target
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    complete = dict(plan, subject_checks=reports,
                    files_sha256={name: digest(out/name) for name in ['signals.npy','rows.jsonl','plan.json']})
    (out/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
    return complete


class PreparedSignalStore:
    """Feed verified, already mapped 2a signals to the unchanged encoder interface."""
    def __init__(self, folder):
        from .leaf_runtime import CONFIG, digest
        from .signal_data import ROOT as SIGNAL_ROOT
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder/'complete.json').read_text())
        assert self.metadata['dataset'] == 'BNCI2014001' and self.metadata['fold'] == CONFIG['fold']
        for name, expected in self.metadata['files_sha256'].items():
            assert digest(self.folder/name) == expected, f'Prepared cache changed: {name}'
        for split, expected in self.metadata['manifests_sha256'].items():
            assert digest(SIGNAL_ROOT/f"fold{CONFIG['fold']}"/f'{split}.jsonl') == expected
        records = [json.loads(s) for s in (self.folder/'rows.jsonl').read_text().splitlines()]
        self.rows = {r['id']:r for r in records}
        assert len(self.rows) == len(records) == self.metadata['n']
        assert [r['prepared_index'] for r in records] == list(range(len(records)))
        self.signals = np.load(self.folder/'signals.npy', mmap_mode='r')
        assert self.signals.shape == (len(records),65,800)

    def batch(self, rows):
        assert rows and {r['dataset'] for r in rows} == {'BNCI2014001'}
        indices = []
        for row in rows:
            saved = self.rows[row['id']]
            assert all(saved[key] == row[key] for key in ['label','subject','split','source_index','task'])
            indices.append(saved['prepared_index'])
        values = np.asarray(self.signals[indices], dtype=np.float32)
        return [(rows[0]['task'], list(range(len(rows))), [values])]
