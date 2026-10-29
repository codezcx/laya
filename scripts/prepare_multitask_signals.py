"""Prepare all frozen records; no re-splitting, no target-dependent normalization."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.benchmark_data import catalog, load_pickle
from eeg_align.multitask_data import locked_records, window_positions, write_json
from scripts.prepare_multitask_splits import DEFINITIONS, montage_channels, extract_methods


def metadata_events(name, row, raw):
    if name in {'tuab', 'tuev', 'adftd'}:
        return json.loads(row['label'])
    if name == 'workload':
        return [('background' if int(row['session']) == 1 else 'arithmetic', 0, -1)]
    if name == 'motor_mv_img':
        labels = {'T1': 'left', 'T2': 'right'} if int(row['session']) in [4, 8, 12] else {'T1': 'both_fist', 'T2': 'foot'}
        return [(labels[d], round(t * 1000), round((t + 4) * 1000)) for t, d in zip(raw.annotations.onset, raw.annotations.description) if d in labels]
    if name == 'hmc':
        path = Path(row['path']).with_name(Path(row['path']).stem + '_sleepscoring.txt')
        frame = pd.read_csv(path, sep=r'\s*,\s*', engine='python')
        begin = frame.index[frame['Annotation'] == 'Lights off'].item() + 1
        end = frame.index[frame['Annotation'] == 'Lights on'].item()
        frame = frame.iloc[begin:end]
        return [(r['Annotation'].split(' ')[-1], round(r['Recording onset'] * 1000), round((r['Recording onset'] + 30) * 1000)) for r in frame.to_dict('records') if r['Duration'] >= 30]
    if name in {'seed', 'seed_vii'}:
        relative, cls = DEFINITIONS[name][:2]
        obj = extract_methods('data/dataset/' + relative, cls, {'_load_meta_info'})()
        obj._load_meta_info()
        if name == 'seed':
            times, labels = obj.time_meta.T, obj.label_meta[0]
        else:
            labels = obj.label_meta[int(row['session']) - 1]
            stem = Path(row['path']).stem
            if stem in {'9_20221111_3', '14_20221015_1'}:
                triggers = pd.read_csv(Path(row['path']).parents[1] / 'save_info' / (stem + '_trigger_info.csv'), header=None, names=['event', 'datetime'])
                origin = '2022-11-11 14:01:27' if stem == '9_20221111_3' else '2022-10-15 14:25:34'
                onsets = (pd.to_datetime(triggers['datetime']) - pd.Timestamp(origin)).dt.total_seconds().to_numpy()
            else:
                onsets = raw.annotations.onset
            if len(onsets) < 40:
                raise ValueError('SEED-VII missing trigger pairs')
            times = onsets[:40].reshape(20, 2)
        return [(DEFINITIONS[name][5][int(y)], round(a * 1000), round(b * 1000)) for y, (a, b) in zip(labels, times)]
    raise ValueError(name)


def filter_raw(raw, name):
    """Upstream continuous filtering, with explicit LEAF 200 Hz resampling."""
    hz = raw.info['sfreq']
    low, high = (.5, 45.) if name == 'bcic_2a' else (.1, 100.)
    length = round(3.3 / min(max(low * .25, 2), low) * hz)
    if raw.n_times > length:
        raw.filter(low, high if hz > high * 2 else None, verbose=False, n_jobs=1)
    notch = 60 if name in {'tuab', 'tuev', 'motor_mv_img'} else 50
    freqs = np.arange(notch, hz / 2, notch)
    if len(freqs):
        raw.notch_filter(freqs, verbose=False, n_jobs=1)
    raw.resample(200, verbose=False, n_jobs=1)
    return raw


def extract_2a(row):
    import mne
    from scipy.io import loadmat
    channels = ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ']
    arrays, entries = [], []
    runs = np.atleast_1d(loadmat(row['path'], squeeze_me=True, struct_as_record=False)['data'])
    for run_index, run in enumerate(runs):
        labels = np.atleast_1d(run.y).astype(int)
        trials = np.atleast_1d(run.trial).astype(int)
        if not labels.size:
            continue
        raw = mne.io.RawArray(run.X[:, :22].T * 1e-6, mne.create_info(channels, float(run.fs), 'eeg'), verbose=False)
        # BNCI MAT trial marks trial start; cue occurs two seconds later.
        # Four seconds from cue, same event-anchored active window as benchmark.
        events = [(DEFINITIONS['bcic_2a'][5][y - 1], round(((t - 1) / run.fs + 2) * 1000), round(((t - 1) / run.fs + 6) * 1000)) for t, y in zip(trials, labels)]
        raw = filter_raw(raw, 'bcic_2a')
        windows, skipped = window_positions(raw.n_times, events, 4, DEFINITIONS['bcic_2a'][5])
        if skipped:
            raise ValueError(f'2a event bounds: {skipped}')
        x = raw.get_data().astype(np.float32)
        arrays.extend(x[:, w['start']:w['stop']] for w in windows)
        entries.extend(dict(w, run=run_index) for w in windows)
    if len(arrays) != 288:
        raise ValueError(f'2a expected 288 trials, got {len(arrays)}')
    return np.stack(arrays), channels, entries, []


def extract_record(name, row):
    import mne
    mne.set_log_level('ERROR')
    if name == 'bcic_2a':
        return extract_2a(row)
    path = Path(row['path'])
    readers = {'.edf': mne.io.read_raw_edf, '.cnt': mne.io.read_raw_cnt, '.set': mne.io.read_raw_eeglab}
    # Six SEED files contain bogus positive sample counts; trust the event-table
    # byte offset instead. All local SEED CNT files are below the 2 GiB limit.
    kwargs = {'data_format': 'int32', 'recompute_n_samples': True} if name == 'seed' else {}
    raw = readers[path.suffix.lower()](path, preload=False, verbose=False, **kwargs)
    events = metadata_events(name, row, raw)
    required = montage_channels(name)
    # CNT headers differ only in capitalization from the upstream converted SET.
    lookup = {c.upper(): c for c in raw.ch_names}
    raw.pick([lookup[c.upper()] for c in required]).load_data()
    if name == 'motor_mv_img':
        raw.set_eeg_reference('average', verbose=False)
    raw = filter_raw(raw, name)
    channels = [c.replace('EEG ', '').replace('-REF', '').replace('.', '').upper() for c in raw.ch_names]
    # HMC remains bipolar; the online adapter records its minimum-norm limitation.
    windows, skipped = window_positions(raw.n_times, events, DEFINITIONS[name][4], DEFINITIONS[name][5], name not in {'tuev', 'seed_vii'})
    x = raw.get_data().astype(np.float32)
    if not windows:
        raise ValueError('No valid epochs in allocated record')
    return np.stack([x[:, w['start']:w['stop']] for w in windows]), channels, windows, skipped


def worker(job):
    name, row, output = job
    output = Path(output)
    done = output.with_suffix('.json')
    source = Path(row['path'])
    stat = source.stat()
    identity = dict(record_id=row['record_id'], source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns, split=row['split'], subject=str(row['subject']), code_sha256=digest(__file__))
    if stat.st_size != int(row['size_bytes']) or stat.st_mtime_ns != int(row['mtime_ns']):
        raise ValueError('Source changed since split allocation')
    if done.exists():
        saved = json.loads(done.read_text())
        if saved['identity'] != identity or not output.exists():
            raise ValueError(f'Incompatible cached record: {output}')
        return saved
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        x, channels, windows, skipped = extract_record(name, row)
    if not np.isfinite(x).all() or any(np.std(v) == 0 for v in x):
        raise ValueError('Invalid finite/nonconstant signal check')
    np.save(output, x.astype(np.float32))
    entries = [dict(array=output.name, row=i, label=w['label'], subject=str(row['subject']), split=row['split'], record_id=row['record_id'], channels=json.dumps(channels), hz=200, samples=x.shape[-1], event=w['event'], start=w['start']) for i, w in enumerate(windows)]
    saved = dict(identity=identity, rows=entries, skipped_events=skipped, warnings=sorted(set(str(w.message) for w in caught)), n=len(entries), bytes=output.stat().st_size)
    write_json(done, saved)
    return saved


def prepare_extra(name, manifest, records, folder):
    if name == 'Nakanishi2015':
        arrays, entries, channels = [], [], None
        for row in records.to_dict('records'):
            with open(row['path'], 'rb') as handle:
                value = pickle.load(handle)
            if channels is None:
                channels = list(value['eeg'])
            if list(value['eeg']) != channels or int(value['freq']) != 200:
                raise ValueError('Nakanishi channel/rate inconsistency')
            signal = np.stack([value['eeg'][c] for c in channels])
            if signal.shape[-1] not in (829, 830, 831):
                raise ValueError('Unexpected Nakanishi epoch length')
            arrays.append(np.pad(signal, ((0, 0), (0, 831 - signal.shape[-1]))))
            entries.append(dict(row, label=round((float(value['label']) - 9.25) / .5)))
        x = np.stack(arrays).astype(np.float32)
        np.save(folder / 'epochs.npy', x)
        hz = 200
        samples = x.shape[-1]
    else:
        if digest(manifest['source']) != manifest['source_sha256']:
            raise ValueError('Extra source content changed')
        data = load_pickle(manifest['source'])
        x = data.eeg_data
        if tuple(x.shape) != tuple(manifest['source_shape']):
            raise ValueError('Extra source shape changed')
        channels, hz, samples = list(data.channel_names), float(data.sampling_rate), x.shape[-1]
        entries = records.to_dict('records')
        mm = np.lib.format.open_memmap(folder / 'epochs.npy', mode='w+', dtype=np.float32, shape=x.shape)
        for start in range(0, len(x), 128):
            part = np.asarray(x[start:start + 128], dtype=np.float32)
            if not np.isfinite(part).all():
                raise ValueError('Nonfinite extra EEG')
            mm[start:start + len(part)] = part
        mm.flush()
        del mm
        for i, row in enumerate(entries):
            if int(row['source_row']) != i or int(np.asarray(data.labels).reshape(-1)[i]) != int(row['label']) or str(data.subject_ids[i]) != str(row['subject']):
                raise ValueError('Cached epoch identity mismatch')
    rows = [dict(array='epochs.npy', row=i, label=int(r['label']), subject=str(r['subject']), split='train', record_id=r['record_id'], channels=json.dumps(channels), hz=hz, samples=samples, event=i, start=0) for i, r in enumerate(entries)]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--audit-one', action='store_true', help='One TRAIN record per dataset; separate cache only')
    args = parser.parse_args()
    cache = Path(CONFIG['cache_root'] + ('_audit_' + digest(__file__)[:8] if args.audit_one else ''))
    cache.mkdir(parents=True, exist_ok=True)
    for task in catalog():
        name = task['task_id']
        if args.datasets and name not in args.datasets:
            continue
        manifest, records = locked_records(name)
        folder = cache / name
        folder.mkdir(exist_ok=True)
        complete_path = folder / 'complete.json'
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete['records_sha256'] != manifest['records_sha256'] or complete['code_sha256'] != digest(__file__) or complete['index_sha256'] != digest(folder / 'index.csv'):
                raise ValueError(f'Incompatible completed cache: {name}')
            print('CACHED', name, complete['counts'], flush=True)
            continue
        print('PREPARING', name, len(records), flush=True)
        if task['role'] == 'train_only':
            rows = prepare_extra(name, manifest, records, folder)
            reports = []
        else:
            if args.audit_one:
                records = records[records['split'] == 'train'].iloc[:1]
            jobs = [(name, r, str(folder / f'record_{i:05d}.npy')) for i, r in enumerate(records.to_dict('records'))]
            reports = []
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
                futures = {executor.submit(worker, job): i for i, job in enumerate(jobs)}
                results = {}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    if len(results) % 20 == 0 or len(results) == len(jobs):
                        print('RECORDS', name, len(results), '/', len(jobs), flush=True)
                        write_json(cache / 'progress.json', dict(dataset=name, prepared=len(results), total=len(jobs)))
                reports = [results[i] for i in range(len(jobs))]
            rows = [row for report in reports for row in report['rows']]
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / 'index.csv', index=False)
        counts = {split: dict(epochs=len(df), subjects=df['subject'].nunique(), labels={str(k): int(v) for k, v in df['label'].value_counts().items()}) for split, df in frame.groupby('split')}
        write_json(complete_path, dict(dataset=name, counts=counts, records_sha256=manifest['records_sha256'], index_sha256=digest(folder / 'index.csv'), code_sha256=digest(__file__), audit_only=args.audit_one, skipped_events=sum(len(r['skipped_events']) for r in reports), preprocessing='continuous upstream-style filtering; LEAF 200Hz instead of upstream 256Hz; source PKL extras unchanged; epoch-local robust normalization online', known_risks=CONFIG['known_risks_accepted']))
        print('READY', name, counts, flush=True)
        gc.collect()
    print('PREPARATION_DONE', flush=True)


if __name__ == '__main__':
    main()
