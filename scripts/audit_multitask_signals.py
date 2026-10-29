"""Scan every prepared epoch without changing raw data, caches, or locked splits."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.benchmark_data import catalog
from eeg_align.multitask_data import locked_records, write_json
from eeg_align.signal_quality import POLICY, invalid_epochs


def scan_array(path):
    path = Path(path)
    before = path.stat()
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f'Unexpected signal shape: {path}: {array.shape}')
    # At most 32 MiB of input per worker, independent of dataset/window size.
    chunk = max(1, (32 * 1024**2) // max(1, array[0].nbytes))
    invalid = {}
    for start in range(0, len(array), chunk):
        reasons = invalid_epochs(array[start:start + chunk])
        invalid.update({start + int(i): str(reasons[i]) for i in np.flatnonzero(reasons != '')})
    sha = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Signal array modified during audit: {path}')
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=sha,
                stat=[after.st_size, after.st_mtime_ns], invalid=invalid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(CONFIG['quality_root']))
    parser.add_argument('--workers', type=int, default=CONFIG['cpu_workers'])
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(CONFIG['cache_root']).resolve()
    started = time.monotonic()
    code = {p: digest(p) for p in ['scripts/audit_multitask_signals.py', 'eeg_align/signal_quality.py']}
    summary = dict(policy=POLICY, code_sha256=code, cache_root=str(root),
                   split_summary_sha256=CONFIG['split_summary_sha256'], datasets={},
                   training_ready=False, heldout_invalid=0, excluded_training_epochs=0)
    # Process one dataset at a time: bounded task queue and no concurrent PKL loads.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for task in catalog():
            name = task['task_id']
            folder = root / name
            complete = json.loads((folder / 'complete.json').read_text())
            index_sha = digest(folder / 'index.csv')
            manifest, _ = locked_records(name)
            if (complete['audit_only'] or complete['index_sha256'] != index_sha
                    or complete['records_sha256'] != manifest['records_sha256']
                    or complete['code_sha256'] != digest('scripts/prepare_multitask_signals.py')):
                raise ValueError(f'Unverified prepared cache: {name}')
            frame = pd.read_csv(folder / 'index.csv', dtype={'subject': str}, keep_default_na=False)
            arrays = {}
            invalid = {}
            futures = {executor.submit(scan_array, folder / array): array for array in sorted(set(frame['array']))}
            for future in as_completed(futures):
                array = futures[future]
                info = future.result()
                invalid[array] = info.pop('invalid')
                arrays[array] = info
            exclusions, heldout = [], []
            coverage = set()
            for i, row in enumerate(frame.to_dict('records')):
                array, r = row['array'], int(row['row'])
                if not 0 <= r < arrays[array]['shape'][0] or (array, r) in coverage:
                    raise ValueError(f'Invalid or duplicate epoch mapping: {name}:{i}')
                coverage.add((array, r))
                if r in invalid[array]:
                    item = dict(row, index=i, id=f'{name}:{i}', reason=invalid[array][r])
                    (exclusions if row['split'] == 'train' else heldout).append(item)
            if len(coverage) != sum(a['shape'][0] for a in arrays.values()):
                raise ValueError(f'Unindexed signals: {name}')
            excluded = {item['index'] for item in exclusions}
            used = frame.drop(index=sorted(excluded))
            before = frame[frame['split'] == 'train']
            after = used[used['split'] == 'train']
            if (set(before['subject']) != set(after['subject'])
                    or set(before['label']) != set(after['label'])):
                raise ValueError(f'QC would remove a training subject or class: {name}')
            report = dict(dataset=name, index_sha256=index_sha, arrays=arrays,
                          before_counts=dict(Counter(frame['split'])), effective_counts=dict(Counter(used['split'])),
                          exclusions=exclusions, invalid_heldout=heldout,
                          reasons=dict(Counter(i['reason'] for i in exclusions + heldout)),
                          excluded_subject_counts=dict(Counter(i['subject'] for i in exclusions)),
                          excluded_label_counts=dict(Counter(str(i['label']) for i in exclusions)))
            write_json(args.out / f'{name}.json', report)
            summary['datasets'][name] = dict(report_sha256=digest(args.out / f'{name}.json'),
                before_counts=report['before_counts'], effective_counts=report['effective_counts'],
                excluded_train=len(exclusions), invalid_heldout=len(heldout), reasons=report['reasons'])
            summary['heldout_invalid'] += len(heldout)
            summary['excluded_training_epochs'] += len(exclusions)
            write_json(args.out / 'progress.json', dict(completed_datasets=len(summary['datasets']),
                latest=name, elapsed_seconds=time.monotonic() - started, **summary['datasets'][name]))
            print(name, json.dumps(summary['datasets'][name]), flush=True)
    summary['training_ready'] = summary['heldout_invalid'] == 0
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if code != {p: digest(p) for p in code}:
        raise RuntimeError('QC code changed during audit')
    write_json(args.out / 'summary.json', summary)
    print('SUMMARY_SHA256', digest(args.out / 'summary.json'), flush=True)
    if not summary['training_ready']:
        raise RuntimeError('Held-out invalid epochs require explicit review; no automatic removal')


if __name__ == '__main__':
    main()
