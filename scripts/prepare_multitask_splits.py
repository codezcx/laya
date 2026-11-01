"""Freeze local record/subject allocations using pinned xw1216 split methods.

This is a metadata-stage manifest, NOT a preprocessed training dataset. Never
calls upstream preproc(), constructors, downloads or cache-cleaning routines.
Official methods are extracted from byte-verified local source, not reimplemented.
Random splits use the recorded local os.walk order; they are not claimed to be
the authors' unpublished concrete subject lists. All output directories are new.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
MNE_ROOT = BASE / 'mne_data'
NB = Path('/inspire/dataset/neural-bench/v1')
UP = ROOT / 'third_party/EEG-FM-Bench-xw1216'
COMMIT = '325398d7d057ecc1216fb3510d70c16eb60337cc'
SPLITS = ('train', 'valid', 'test')
EXTRAS = ('BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'CHB_MIT',
          'SleepEDF', 'AlexMI', 'Zhou2016', 'Nakanishi2015')
DEFINITIONS = {
    'bcic_2a': ('bcic/bcic_2a.py', 'BCIC2ABuilder', .1, .1, 4,
                ['left', 'right', 'foot', 'tongue']),
    'motor_mv_img': ('motor_mv_img.py', 'MotorMoveImagineBuilder', .1, .1, 4,
                     ['left', 'right', 'both_fist', 'foot']),
    'seed': ('seeds/seed.py', 'SeedBuilder', .067, .067, 4, ['sad', 'neutral', 'happy']),
    'seed_vii': ('seeds/seed_vii.py', 'SeedVIIBuilder', .1, .1, 15,
                 ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Anger', 'Surprise']),
    'tuab': ('tue/tuab.py', 'TuabBuilder', .05, .05, 15, ['normal', 'abnormal']),
    'tuev': ('tue/tuev.py', 'TuevBuilder', .1, .1, 5,
             ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']),
    'hmc': ('hmc.py', 'HMCBuilder', .162, .162, 30, ['W', 'R', 'N1', 'N2', 'N3']),
    'workload': ('workload.py', 'WorkloadBuilder', .14, .14, 4, ['background', 'arithmetic']),
    'adftd': ('adftd.py', 'AdftdBuilder', .15, .15, 10, ['AD', 'FTD', 'CN']),
}
BASE_METHODS = {
    '_divide_all_split_by_sub', '_iterative_greedy_split', '_analyze_split',
    '_multi_label_iterative_stratified_split', '_divide_label_balance_all_split',
    '_divide_balance_test_from_valid', '_milli_sec_to_pts',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def pinned_source(relative):
    expected = subprocess.check_output(['git', '-C', str(UP), 'show', f'{COMMIT}:{relative}'])
    actual = (UP / relative).read_bytes()
    if actual != expected:
        raise RuntimeError(f'Upstream source differs from pinned commit: {relative}')
    return actual.decode()


def extract_methods(relative, class_name, methods, base=object):
    tree = ast.parse(pinned_source(relative))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if {n.name for n in selected} != set(methods):
        raise RuntimeError(f'Missing audited methods in {relative}')
    # A new class containing ONLY the reviewed split functions. No module imports,
    # dataclass evaluation, builder constructors or preproc/cleanup entry points.
    wrapper = ast.ClassDef(name='SplitOnly', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                          keywords=[], body=selected, decorator_list=[])
    mod = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = dict(Parent=base, np=np, pd=pd, math=math, json=json,
                     ndarray=np.ndarray, DataFrame=pd.DataFrame, logger=logging.getLogger('splits'))
    exec(compile(mod, str(UP / relative), 'exec'), namespace)
    return namespace['SplitOnly']


def splitter(dataset):
    relative, cls, vr, tr, seconds, categories = DEFINITIONS[dataset]
    parent = extract_methods('data/processor/builder.py', 'EEGDatasetBuilder', BASE_METHODS)
    derived = extract_methods('data/dataset/' + relative, cls, {'_divide_split'}, parent)
    obj = derived()
    obj.config = SimpleNamespace(is_finetune=True, is_cross_subject=True, seed=42,
                                 valid_ratio=vr, test_ratio=tr, fs=256, wnd_len=256 * seconds,
                                 persist_drop_last=dataset not in ('tuev', 'seed_vii'),
                                 category=categories)
    obj.split_corr = dict(zip(SPLITS, range(3)))
    return obj


def montage_channels(dataset):
    tree = ast.parse(pinned_source('data/dataset/' + DEFINITIONS[dataset][0]))
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith('Config'))
    value = next(n.value for n in config.body if isinstance(n, ast.AnnAssign)
                 and isinstance(n.target, ast.Name) and n.target.id == 'montage')
    factory = next(k.value for k in value.keywords if k.arg == 'default_factory')
    montages = ast.literal_eval(factory.body)
    if len(montages) != 1:
        raise ValueError('Expected one explicitly reviewed montage')
    return next(iter(montages.values()))


def header_qc(dataset, rows):
    """Match upstream channel/length exclusions where EDF headers are available."""
    if dataset not in ('tuab', 'tuev', 'hmc', 'workload', 'motor_mv_img'):
        return rows, []
    expected = set(montage_channels(dataset))
    def check(row):
        r = dict(row)
        if 'channels' not in r:
            r.update(edf_metadata(Path(r['path'])))
        missing = sorted(expected - set(r['channels']))
        reasons = []
        if missing:
            reasons.append('missing_montage_channels:' + ','.join(missing))
        if r['time'] < DEFINITIONS[dataset][4]:
            reasons.append('shorter_than_official_window')
        if reasons:
            r['exclusion_reason'] = ';'.join(reasons)
        return r
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, rows))
    return ([r for r in checked if 'exclusion_reason' not in r],
            [r for r in checked if 'exclusion_reason' in r])


def allocate(dataset, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f'No input records: {dataset}')
    if df['record_id'].duplicated().any():
        raise ValueError('Duplicate record identifiers')
    state = np.random.get_state()
    try:
        np.random.seed(42)
        result = splitter(dataset)._divide_split(df.copy())
    finally:
        np.random.set_state(state)
    validate(result)
    return result


def validate(df, train_only=False):
    if df.empty or not set(df['split']) <= set(SPLITS):
        raise ValueError('Missing/invalid split allocation')
    if df.groupby('subject')['split'].nunique().max() != 1:
        raise ValueError('Subject appears in multiple splits')
    if df['record_id'].duplicated().any():
        raise ValueError('Record appears more than once')
    if train_only:
        if set(df['split']) != {'train'}:
            raise ValueError('Extra data must be train-only')
    elif set(df['split']) != set(SPLITS):
        raise ValueError('Empty required split')


def walk(root, suffix, accept=lambda p: True):
    # Match the upstream scanner: do not silently sort subjects before sampling.
    for folder, _, files in os.walk(root):
        for name in files:
            path = Path(folder) / name
            if name.endswith(suffix) and accept(path):
                yield path


def edf_metadata(path):
    with path.open('rb') as f:
        header = f.read(256)
        if len(header) != 256:
            raise ValueError('Truncated EDF header')
        channels = int(header[252:256])
        labels = f.read(channels * 16)
    records, duration = int(header[236:244]), float(header[244:252])
    if records < 0 or duration <= 0:
        raise ValueError('EDF duration unavailable in header')
    tokens = header[8:88].decode('ascii', errors='replace').split()
    code = tokens[0] if tokens else ''
    # Never export the raw EDF patient field (potential identifying information).
    # TUH identifiers here are numeric or alphabetic. The TUEV train headers
    # instead contain one repeated slash-prefixed placeholder; do not link it.
    valid_code = code.isalnum() and code.lower() not in (
        '', 'x', 'unknown', 'anonymous', '0', '00000000')
    patient_key = hashlib.sha256(code.encode()).hexdigest() if valid_code else ''
    return dict(time=records * duration, patient_key=patient_key,
                header_sha256=hashlib.sha256(header + labels).hexdigest(),
                channels=[labels[i * 16:(i + 1) * 16].decode('ascii').strip() for i in range(channels)])


def basic(path, subject, session):
    stat = path.stat()
    return dict(path=str(path), subject=subject, session=session, record_id=path.stem,
                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def collect(dataset):
    """Read record identity and split-required metadata, not EEG signal arrays."""
    if dataset == 'bcic_2a':
        paths = list(walk(MNE_ROOT / 'BCIC-IV-2a', '.mat',
                          lambda p: bool(re.fullmatch(r'A\d\d[TE]\.mat', p.name))))
        rows = [basic(p, int(p.stem[1:3]), p.stem[3]) for p in paths]
    elif dataset == 'motor_mv_img':
        paths = walk(MNE_ROOT / 'eegbci/MNE-eegbci-data/files/eegmmidb/1.0.0', '.edf',
                     lambda p: int(p.stem.split('R')[-1]) in (4, 6, 8, 10, 12, 14))
        rows = [basic(p, int(p.stem[1:4]), int(p.stem.split('R')[-1])) for p in paths]
    elif dataset in ('seed', 'seed_vii'):
        root = MNE_ROOT / ('SEED/SEED/SEED_EEG/SEED_RAW_EEG' if dataset == 'seed' else 'SEED-VII/EEG_raw')
        rows = [basic(p, int(p.stem.split('_')[0]), int(p.stem.split('_')[-1]))
                for p in walk(root, '.cnt')]
    elif dataset == 'hmc':
        rows = [basic(p, int(p.stem[2:]), 1) for p in walk(MNE_ROOT / 'HMC/recordings', '.edf',
                                                        lambda p: '_' not in p.stem)]
        for row in rows:
            annotation = Path(row['path']).with_name(row['record_id'] + '_sleepscoring.txt')
            if not annotation.is_file():
                raise ValueError(f'Missing sleep annotations: {row["record_id"]}')
    elif dataset == 'workload':
        rows = [basic(p, p.stem.split('_')[0], int(p.stem.split('_')[1]))
                for p in walk(MNE_ROOT / 'EEGMat', '.edf')]
    elif dataset == 'adftd':
        root = NB / 'Miltiadous2023Dice/download'
        participants = pd.read_csv(root / 'participants.tsv', sep='\t').set_index('participant_id')
        rows = []
        for p in sorted(root.glob('sub-*/eeg/*.set')):  # excludes derivatives, as upstream requires
            subject = p.stem.split('_')[0]
            row = basic(p, subject.removeprefix('sub-'), 1)
            meta = json.loads(p.with_suffix('.json').read_text())
            row['time'] = float(meta['RecordingDuration'])
            label = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}[participants.loc[subject, 'Group']]
            row['label'] = json.dumps([[label, 0, -1]])
            rows.append(row)
    elif dataset in ('tuab', 'tuev'):
        root = NB / ('Lopez2017Tuab/edf' if dataset == 'tuab' else 'Harati2015Tuev/edf')
        paths = list(walk(root, '.edf'))
        def one(p):
            relative = p.relative_to(root).parts
            original_split = relative[0]
            if dataset == 'tuab':
                subject, session, _ = p.stem.split('_')
                row = basic(p, subject, int(session[1:]))
            else:
                if original_split == 'train':
                    subject, session = p.stem.split('_')[-2:]
                    session = int(session)
                else:
                    _, subject, _, session = p.stem.split('_')
                    session = 1 if session == '' else int(session) + 1
                row = basic(p, subject, session)
            # Relative path is needed: TUEV can reuse basenames in source partitions.
            row['record_id'] = '/'.join(relative)
            row.update(edf_metadata(p))
            row['split'] = row['source_split'] = original_split
            if dataset == 'tuab':
                row['label'] = json.dumps([[relative[1], 0, -1]])
            else:
                rec = pd.read_csv(p.with_suffix('.rec'), header=None,
                                  names=['channel', 'start', 'end', 'label'])
                rec = rec[rec.start <= row['time']].copy()
                rec['end'] = rec.end.clip(upper=row['time'])
                labels = DEFINITIONS[dataset][-1]
                row['label'] = json.dumps([[labels[int(r.label) - 1], int(r.start * 1000), int(r.end * 1000)]
                                           for r in rec.itertuples()])
                row['annotation_sha256'] = sha(p.with_suffix('.rec'))
            return row
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(one, paths))
    else:
        raise ValueError(dataset)
    return rows


def summary(df, samples=None):
    result = {}
    for split in SPLITS:
        selected = df[df.split == split]
        people = sorted(map(str, selected.subject.unique()))
        result[split] = dict(subjects=people, n_subjects=len(people), n_records=len(selected))
        if samples is not None:
            result[split]['n_samples'] = int(samples if split == 'train' else 0)
    return result


def extra(dataset, output):
    sys.path.insert(0, str(ROOT))
    from eeg_align.benchmark_data import load_pickle
    if dataset == 'Nakanishi2015':
        source = Path('/inspire/dataset/moabb-processed/v1/EEG-processed/Nakanishi2015')
        rows = []
        for p in walk(source, '.pkl'):
            # Existing trusted local preprocessing products, not downloaded code.
            with p.open('rb') as f:
                d = pickle.load(f)
            row = basic(p, str(d['subject']), p.parent.name)
            row.update(record_id=str(p.relative_to(source)), split='train', label=str(d['label']))
            rows.append(row)
        df = pd.DataFrame(rows)
        validate(df, train_only=True)
        df.to_csv(output / 'records.csv', index=False)
        return dict(role='train_only', counts=summary(df, len(df)), source=str(source),
                    identity_limit='Local trial files; raw recording identity still requires provenance audit')
    source = BASE / 'EEG-FM-Benchmark/datasets/data' / (dataset + '.pkl')
    d = load_pickle(source)
    subjects = np.asarray(d.subject_ids).astype(str)
    labels = np.asarray(d.labels)
    if len(subjects) != len(labels) or len(subjects) != len(d.eeg_data):
        raise ValueError('Signal/subject/label length mismatch')
    df = pd.DataFrame(dict(subject=subjects, record_id=[f'{dataset}:row:{i}' for i in range(len(subjects))],
                           source_row=np.arange(len(subjects)), label=labels, split='train'))
    validate(df, train_only=True)
    df.to_csv(output / 'records.csv', index=False)
    stat = source.stat()
    result = dict(role='train_only', record_unit='cached_epoch_row', counts=summary(df, len(df)), source=str(source),
                  source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                  source_sha256=sha(source), source_shape=list(d.eeg_data.shape),
                  sampling_rate=float(d.sampling_rate), labels=dict(Counter(map(str, labels))),
                  identity_limit='PKL row identifiers are NOT raw recording IDs; source provenance audit pending')
    if dataset == 'Zhou2016':
        result['coverage'] = 'Existing cache contains session 0 only, not all original sessions'
    del d
    gc.collect()
    return result


def cross_tuh_audit(tables):
    groups = defaultdict(list)
    unresolved = 0
    for dataset in ('tuab', 'tuev'):
        if dataset not in tables:
            continue
        for r in tables[dataset].to_dict('records'):
            key = r.get('patient_key')
            if not key:
                unresolved += 1
                continue
            groups[key].append((dataset, str(r['subject']), r['split']))
    matches, conflicts = [], []
    for key, values in groups.items():
        if len({v[0] for v in values}) < 2:
            continue
        item = dict(key_sha256=key, allocations=[dict(dataset=d, subject=s, split=p)
                                               for d, s, p in sorted(set(values))])
        matches.append(item)
        if len({v[2] for v in values}) > 1:
            conflicts.append(item)
    shared_names, name_conflicts = [], []
    if all(d in tables for d in ('tuab', 'tuev')):
        a, b = (tables[d] for d in ('tuab', 'tuev'))
        # These are public pseudonymous filename IDs, not raw EDF patient fields.
        for subject in sorted(set(a.subject) & set(b.subject)):
            assignments = [dict(dataset=d, subject=str(subject), split=str(t[t.subject == subject].split.iloc[0]),
                                n_records=int((t.subject == subject).sum()))
                           for d, t in (('tuab', a), ('tuev', b))]
            shared_names.append(assignments)
            if len({r['split'] for r in assignments}) > 1:
                name_conflicts.append(assignments)
    return dict(header_code_matches=matches, role_conflicts=conflicts,
                shared_filename_ids=shared_names, candidate_filename_role_conflicts=name_conflicts,
                records_without_usable_code=unresolved,
                limitation='Header code matching is a screening check, not a verified cross-version patient map; '
                           'zero matches does not prove disjoint patients. Matching filename IDs are potential '
                           'shared patients until cross-version namespace provenance is verified.')


def hmc_coverage():
    root = MNE_ROOT / 'HMC'
    source = root / 'RECORDS'
    expected = [r.strip() for r in source.read_text().splitlines() if r.strip()]
    missing = [r for r in expected if not (root / r).is_file()]
    return dict(records_file_sha256=sha(source), expected_recordings=len(expected),
                available_recordings=len(expected) - len(missing), missing_recordings=missing)


def verify_complete_hmc(report_path):
    """Require the official complete release, not a live progress file or subset."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    root = MNE_ROOT / 'HMC'
    if (report.get('status') != 'complete' or report.get('recordings') != 151
            or report.get('annotation_txt') != 151 or report.get('annotation_edf') != 151
            or report.get('official_files') != 457 or report.get('final_verified_files') != 457
            or report.get('final_mismatches') != [] or report.get('original_files_changed') != []
            or Path(report.get('dataset_root', '')).resolve() != root.resolve()):
        raise ValueError('HMC download is not a verified complete 151-record release')
    manifest = json.loads(report_path.with_name('official_manifest.json').read_text())
    if (manifest['url'] != 'https://physionet.org/files/hmc-sleep-staging/1.1/SHA256SUMS.txt'
            or manifest['manifest_sha256'] != sha(root / 'SHA256SUMS.txt')):
        raise ValueError('Official HMC checksum manifest identity changed')
    # Do not trust a stale complete.json: re-hash all listed data and annotations.
    entries = {}
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip('*')
        target = root / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or target.is_symlink() or not target.resolve().is_relative_to(root.resolve())
                or not re.fullmatch(r'[0-9a-f]{64}', digest) or relative in entries):
            raise ValueError('Unsafe or duplicate checksum entry')
        entries[relative] = digest
    if len(entries) != 457 or entries != manifest['files']:
        raise ValueError('HMC checksum entries disagree with official saved manifest')
    expected = {n for n in entries if re.fullmatch(r'recordings/SN\d+\.edf', n)}
    observed = {str(p.relative_to(root)) for p in (root / 'recordings').glob('SN*.edf')
                if re.fullmatch(r'SN\d+\.edf', p.name)}
    coverage = hmc_coverage()
    if (len(expected) != 151 or expected != observed
            or coverage['expected_recordings'] != 151 or coverage['missing_recordings']):
        raise ValueError('HMC source recording set is incomplete or includes extra records')
    names = list(entries)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda name: sha(root / name), names))
    if any(actual[i] != entries[name] for i, name in enumerate(names)):
        raise ValueError('HMC file checksum mismatch after download')
    return dict(complete_report_sha256=sha(report_path),
                official_manifest_sha256=manifest['manifest_sha256'],
                verified_files=len(entries), verified_recordings=len(expected))


def wait_for_complete_hmc(report_path, timeout_seconds=24 * 3600):
    """Wait on the actual downloader, never mistake a stale progress file for liveness."""
    report_path = Path(report_path).resolve()
    progress_path = report_path.with_name('progress.json')
    expected_script = report_path.with_name('download.py')
    started = time.monotonic()
    while not report_path.exists():
        state = json.loads(progress_path.read_text())
        pid = state.get('downloader_pid')
        if state['status'] in {'incomplete', 'verification_failed', 'blocked_existing_mismatch'}:
            raise RuntimeError(f"HMC downloader terminal state: {state['status']}")
        if not isinstance(pid, int) or pid <= 0:
            raise RuntimeError('HMC downloader has no verifiable process handle')
        proc = Path('/proc') / str(pid)
        try:
            arguments = proc.joinpath('cmdline').read_bytes().split(b'\0')
            cwd = proc.joinpath('cwd').resolve(strict=True)
            scripts = [(cwd / a.decode()).resolve() for a in arguments if a.endswith(b'download.py')]
            if expected_script not in scripts:
                raise RuntimeError('HMC downloader process handle no longer matches')
        except FileNotFoundError as exc:
            # Completion can race process exit; recheck before calling it stopped.
            if report_path.exists():
                break
            raise RuntimeError('HMC downloader exited without a completion report') from exc
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError('HMC preparation wait exceeded 24 hours; downloader not stopped')
        print('WAIT_HMC', json.dumps(dict(pid=pid, status=state['status'],
            downloaded=len(state.get('downloaded', [])), pending=state.get('pending_files'))), flush=True)
        time.sleep(30)


def retain_other_allocations(overall, output, base, expected_sha):
    """Merge ONLY new HMC allocation; copy every other frozen metadata file."""
    base, output = Path(base), Path(output)
    if sha(base / 'summary.json') != expected_sha:
        raise ValueError('Base split summary changed')
    previous = json.loads((base / 'summary.json').read_text())
    if previous['upstream_commit'] != COMMIT or set(overall['datasets']) != {'hmc'}:
        raise ValueError('HMC refresh requires the same pinned upstream and one changed dataset')
    retained = {}
    for name, manifest in previous['datasets'].items():
        if name == 'hmc':
            continue
        source = base / name
        if (json.loads((source / 'manifest.json').read_text()) != manifest
                or sha(source / 'records.csv') != manifest['records_sha256']):
            raise ValueError(f'Base allocation changed: {name}')
        shutil.copytree(source, output / name)
        if sha(output / name / 'records.csv') != manifest['records_sha256']:
            raise ValueError(f'Copied allocation changed: {name}')
        overall['datasets'][name] = manifest
        retained[name] = manifest['records_sha256']
    overall['tuh_overlap_audit'] = previous['tuh_overlap_audit']
    overall['blocking_decisions'] = [item for item in previous['blocking_decisions']
                                     if not item.startswith('HMC missing recordings:')]
    overall['refresh'] = dict(base_summary_sha256=expected_sha, changed_datasets=['hmc'],
                              retained_records_sha256=retained,
                              warm_start_from_old_trained_checkpoint=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=[*DEFINITIONS, *EXTRAS],
                        default=[*DEFINITIONS, *EXTRAS])
    parser.add_argument('--refresh-hmc-from', type=Path,
                        help='Reallocate complete HMC only; preserve every other frozen allocation')
    parser.add_argument('--base-summary-sha256')
    parser.add_argument('--hmc-download-report', type=Path)
    parser.add_argument('--wait-for-hmc-download', action='store_true',
                        help='Wait on the verified live downloader before full integrity verification')
    args = parser.parse_args()
    if args.wait_for_hmc_download and not args.refresh_hmc_from:
        parser.error('--wait-for-hmc-download requires --refresh-hmc-from')
    download_proof = None
    if args.refresh_hmc_from:
        if not args.base_summary_sha256 or not args.hmc_download_report:
            parser.error('HMC refresh requires --base-summary-sha256 and --hmc-download-report')
        if sha(args.refresh_hmc_from / 'summary.json') != args.base_summary_sha256:
            raise ValueError('Base split summary changed')
        if args.wait_for_hmc_download:
            wait_for_complete_hmc(args.hmc_download_report)
        download_proof = verify_complete_hmc(args.hmc_download_report)
        args.datasets = ['hmc']
    args.output.mkdir(parents=True, exist_ok=False)
    overall = dict(schema='eegjev-record-splits-1', upstream_commit=COMMIT, seed=42,
                   status='metadata_allocation_only', training_ready=False, datasets={}, errors={},
                   excluded=['BNCI2014004'],
                   aliases={'BNCI2014001': 'bcic_2a', 'BNCI2014001Session': 'bcic_2a',
                            'EEGMAT': 'workload', 'SEED': 'seed', 'TUAB': 'tuab'},
                   limitations=['No waveform preprocessing or model fitting performed',
                                'Benchmark record/channel/event QC pending; allocations must not be silently redrawn',
                                'Cross-corpus participant identity not globally established',
                                'Published foundation checkpoint pretraining overlap unresolved'])
    tables = {}
    for dataset in args.datasets:
        print(f'Preparing {dataset}', flush=True)
        output = args.output / dataset
        output.mkdir()
        try:
            if dataset in DEFINITIONS:
                raw_rows = collect(dataset)
                rows, rejected = header_qc(dataset, raw_rows)
                df = allocate(dataset, rows)
                df['canonical_dataset'] = dataset
                df['allocation_row'] = np.arange(len(df))
                df.to_csv(output / 'records.csv', index=False)
                pd.DataFrame(rejected, columns=sorted(set().union(*(r.keys() for r in rejected)))
                             if rejected else ['record_id', 'exclusion_reason']).to_csv(
                                 output / 'excluded_records.csv', index=False)
                source_file = 'data/dataset/' + DEFINITIONS[dataset][0]
                result = dict(role='benchmark', record_unit='source_recording', counts=summary(df),
                              n_scanned_records=len(raw_rows), n_header_qc_excluded=len(rejected),
                              excluded_records_sha256=sha(output / 'excluded_records.csv'),
                              source_sha256={p: sha(UP / p) for p in ('data/processor/builder.py', source_file)},
                              subject_input_order=list(dict.fromkeys(str(r['subject']) for r in rows)),
                              scanner='local os.walk encounter order, except sorted ADFTD BIDS subject files',
                              qc_stage='identity and split-required metadata only; not upstream full preprocessing')
                if dataset == 'hmc':
                    result['coverage'] = hmc_coverage()
                    if result['coverage']['missing_recordings']:
                        result['allocation_status'] = 'provisional_local_subset_not_complete_HMC'
                if dataset in ('tuab', 'tuev'):
                    tables[dataset] = df
            else:
                result = extra(dataset, output)
            result['records_sha256'] = sha(output / 'records.csv')
            write_json(output / 'manifest.json', result)
            overall['datasets'][dataset] = result
            print(dataset, {s: v['n_subjects'] for s, v in result['counts'].items()}, flush=True)
        except Exception as e:
            # Preserve partial successful datasets and make all failures explicit.
            overall['errors'][dataset] = f'{type(e).__name__}: {e}'
            print(f'FAILED {dataset}: {type(e).__name__}: {e}', flush=True)
    overall['tuh_overlap_audit'] = cross_tuh_audit(tables)
    overall['blocking_decisions'] = []
    if overall['tuh_overlap_audit']['candidate_filename_role_conflicts']:
        overall['blocking_decisions'].append('Potential TUAB/TUEV train/held-out patient overlap: '
                                             'verify mapping and approve cross-corpus quarantine policy before training')
    if overall['datasets'].get('hmc', {}).get('coverage', {}).get('missing_recordings'):
        overall['blocking_decisions'].append('HMC missing recordings: complete download or explicitly approve subset/defer')
    if args.refresh_hmc_from and not overall['errors']:
        counts = overall['datasets']['hmc']['counts']
        if {s: counts[s]['n_subjects'] for s in SPLITS} != {'train': 103, 'valid': 24, 'test': 24}:
            raise ValueError('Full HMC allocation must contain 103/24/24 disjoint subjects')
        retain_other_allocations(overall, args.output, args.refresh_hmc_from, args.base_summary_sha256)
        overall['refresh']['download_proof'] = download_proof
    overall['preparation_code_sha256'] = sha(Path(__file__))
    write_json(args.output / 'summary.json', overall)
    print('Saved metadata allocations; training_ready=false (QC/identity/preprocessing gates pending)', flush=True)
    if overall['errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
