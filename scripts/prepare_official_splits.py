"""Audit original benchmark data and export its unmodified Cross split masks.

No EEG normalization, sample filtering, subject regrouping, fitting or scoring.
Missing data and unusable folds are recorded, not silently dropped. Split RNG is
reset before each dataset/seed so every later model can consume identical masks.
"""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

UPSTREAM_COMMIT = 'b36719588abec452e4c2b564f25ede1c0b1c03bc'
DATASETS = ['BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'BNCI2014008',
            'BNCI2014009', 'CHB_MIT', 'TUAB', 'SleepEDF', 'SEED', 'EEGMAT',
            'Nakanishi2015', 'SEED_VIG', 'ThingsEEG2']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source(path, schema):
    class TrustedLocalUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'utils.EEGDataLoader' and name == 'EEGData':
                return schema.EEGData
            return super().find_class(module, name)
    with path.open('rb') as handle:
        return TrustedLocalUnpickler(handle).load()


def mask_summary(data, mask):
    indices = np.flatnonzero(mask)
    subjects = np.asarray(data.subject_ids)[mask]
    result = dict(n=len(indices), subjects=sorted(set(map(str, subjects)), key=float))
    if getattr(data, 'labels', None) is not None:
        result['labels'] = dict(Counter(map(str, np.asarray(data.labels)[mask])))
    if getattr(data, 'usage', None) is not None:
        result['usage'] = dict(Counter(map(str, np.asarray(data.usage)[mask])))
    return indices, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == UPSTREAM_COMMIT
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', '--',
                    'utils/dataset_split.py', 'utils/EEGDataLoader.py'], check=True)
    args.out.mkdir(parents=True, exist_ok=False)
    schema = import_file('official_eeg_schema', root / 'utils/EEGDataLoader.py')
    splitter = import_file('official_eeg_splitter', root / 'utils/dataset_split.py')
    report = dict(upstream_commit=commit, split_code_sha256=sha256(root / 'utils/dataset_split.py'),
        script_sha256=sha256(__file__), seeds=args.seeds, mode='Cross', datasets={},
        split_rng='np.random.seed(seed) immediately before unmodified official splitter; common masks for all models',
        scope='data/protocol preparation only; no models trained or evaluated',
        restrictions=['No additional validation subject', 'Original usage honored',
                      'No sample exclusions or subject regrouping',
                      'Not a claim of matching unavailable paper-specific fold manifests'])
    started = time.monotonic()
    for name in DATASETS:
        source = args.data_root / f'{name}.pkl'
        if not source.is_file():
            report['datasets'][name] = dict(status='missing_source', expected_path=str(source.resolve()))
            write_json(args.out / 'progress.json', report)
            print('MISSING', name, flush=True)
            continue
        print('LOADING', name, flush=True)
        before = source.stat()
        data = load_source(source, schema)
        subjects = np.asarray(data.subject_ids)
        assert len(subjects) == len(data.eeg_data)
        method = getattr(data, 'split_method', 'LOSO')
        folder = args.out / name
        folder.mkdir()
        info = dict(status='prepared', source=str(source.resolve()), source_size=before.st_size,
            source_mtime_ns=before.st_mtime_ns, source_sha256=sha256(source),
            samples=len(subjects), subjects=len(np.unique(subjects)),
            task_type=data.dataset_type, split_method=method, signal_shape=list(data.eeg_data.shape),
            sampling_rate=float(data.sampling_rate),
            usage=None if getattr(data, 'usage', None) is None else dict(Counter(map(str, data.usage))),
            folds=[], issues=[])
        for seed in args.seeds:
            np.random.seed(seed)
            splits = splitter.split_dataset_cross(data, method)
            for fold, split in enumerate(splits):
                train_mask, test_mask = split['train_mask'], split['test_mask']
                assert not np.any(train_mask & test_mask)
                train, train_info = mask_summary(data, train_mask)
                test, test_info = mask_summary(data, test_mask)
                assert not set(train_info['subjects']) & set(test_info['subjects'])
                if not len(train) or not len(test):
                    info['issues'].append(dict(seed=seed, fold=fold, error='empty_train_or_test'))
                target = folder / f'seed{seed}_fold{fold:03d}.npz'
                np.savez_compressed(target, train=train, test=test)
                info['folds'].append(dict(seed=seed, fold=fold, train=train_info, test=test_info,
                    indices_file=str(target.relative_to(args.out)), indices_sha256=sha256(target)))
            del splits
        after = source.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        if info['issues']:
            info['status'] = 'prepared_with_issues'
        write_json(folder / 'manifest.json', info)
        report['datasets'][name] = {k:v for k,v in info.items() if k != 'folds'}
        report['datasets'][name]['fold_runs'] = len(info['folds'])
        report['datasets'][name]['manifest_sha256'] = sha256(folder / 'manifest.json')
        write_json(args.out / 'progress.json', report)
        print('PREPARED', name, info['samples'], info['subjects'], len(info['folds']), info['issues'], flush=True)
        del data, subjects, info
        gc.collect()
    report['elapsed_s'] = time.monotonic() - started
    report['all_sources_available'] = all(d['status'] != 'missing_source' for d in report['datasets'].values())
    report['all_folds_usable'] = report['all_sources_available'] and all(not d.get('issues') for d in report['datasets'].values())
    report['training_started'] = False
    write_json(args.out / 'complete.json', report)
    print('AUDIT_COMPLETE', json.dumps({k:report[k] for k in ['all_sources_available', 'all_folds_usable', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
