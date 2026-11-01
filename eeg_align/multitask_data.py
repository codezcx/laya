"""Frozen record allocations and bounded mmap caches, shared by current controls."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_data import catalog
from .runtime import CONFIG, digest
from .signal_quality import load_quality, excluded_training_indices
from .split_policy import validate_split_frame


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def locked_records(name):
    root = Path(CONFIG['split_root'])
    if digest(root / 'summary.json') != CONFIG['split_summary_sha256']:
        raise ValueError('Split summary changed')
    manifest = json.loads((root / name / 'manifest.json').read_text())
    if manifest != json.loads((root / 'summary.json').read_text())['datasets'][name]:
        raise ValueError('Dataset manifest differs from frozen summary')
    if digest(root / name / 'records.csv') != manifest['records_sha256']:
        raise ValueError(f'Locked rows changed: {name}')
    rows = pd.read_csv(root / name / 'records.csv', dtype={'subject': str}, keep_default_na=False)
    validate_split_frame(name, manifest, rows)
    return manifest, rows


def window_positions(length, events, seconds, categories, drop_last=True, hz=200):
    """Pinned upstream full-window and end-aligned tail semantics, with audit."""
    size = int(seconds * hz)
    windows, skipped = [], []
    for event, (label, start_ms, end_ms) in enumerate(events):
        if label not in categories:
            skipped.append(dict(event=event, reason='unknown_label', label=label))
            continue
        start = round(start_ms * hz / 1000)
        end = length if end_ms < 0 else round(end_ms * hz / 1000)
        if start < 0 or end > length or start >= end or length < size:
            skipped.append(dict(event=event, reason='out_of_record_bounds'))
            continue
        count, remainder = divmod(end - start, size)
        positions = [start + i * size for i in range(count)]
        if not drop_last and remainder:
            positions.append(max(0, end - size) if end >= size else start)
        for pos in positions:
            if not 0 <= pos <= length - size:
                raise ValueError('Invalid end-aligned window')
            windows.append(dict(start=pos, stop=pos + size, label=categories.index(label), event=event))
    return windows, skipped


class TrialStore:
    def __init__(self):
        self.tasks = catalog()
        self.rows = {}
        self.meta = {}
        self.quality = load_quality(CONFIG)
        if set(self.quality) != {t['task_id'] for t in self.tasks}:
            raise ValueError('Signal QC must cover every configured dataset')
        self.excluded = {}
        root = Path(CONFIG['cache_root'])
        for tid, task in enumerate(self.tasks):
            name = task['task_id']
            folder = root / name
            complete = json.loads((folder / 'complete.json').read_text())
            if complete['audit_only']:
                raise ValueError('Audit cache cannot be used for training')
            if complete['code_sha256'] != digest('scripts/prepare_multitask_signals.py'):
                raise ValueError('Signal preprocessing code changed')
            manifest, _ = locked_records(name)
            if complete['records_sha256'] != manifest['records_sha256']:
                raise ValueError('Signal cache belongs to another split')
            if complete['index_sha256'] != digest(folder / 'index.csv'):
                raise ValueError('Epoch index changed')
            df = pd.read_csv(folder / 'index.csv', dtype={'subject': str}, keep_default_na=False)
            if not set(df['split']) <= {'train', 'valid', 'test'}:
                raise ValueError('Invalid split')
            if task['role'] == 'train_only' and set(df['split']) != {'train'}:
                raise ValueError('Extra data used for evaluation')
            validate_split_frame(name, manifest, df, windows=True)
            self.rows[tid] = [dict(r, task=tid, id=f'{name}:{i}') for i, r in enumerate(df.to_dict('records'))]
            self.meta[tid] = complete
            self.excluded[tid] = excluded_training_indices(df, self.quality[name]['exclusions'])
        self.pools = {}
        for tid, rows in self.rows.items():
            pool = defaultdict(lambda: defaultdict(list))
            for i, row in enumerate(rows):
                if row['split'] == 'train' and i not in self.excluded[tid]:
                    pool[int(row['label'])][row['subject']].append(i)
            if set(pool) != set(range(len(self.tasks[tid]['classes']))):
                raise ValueError(f'Missing training classes: {self.tasks[tid]["task_id"]}')
            self.pools[tid] = pool

    def sample(self, step, batch, seed):
        cycle, position = divmod(step - 1, len(self.tasks))
        rng = np.random.default_rng(np.random.SeedSequence([seed, cycle]))
        tid = int(rng.permutation(len(self.tasks))[position])
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, tid]))
        pool = self.pools[tid]
        classes = rng.permutation(sorted(pool))
        selected = []
        for i in range(batch):
            label = int(classes[i % len(classes)])
            subjects = sorted(pool[label])
            subject = subjects[int(rng.integers(len(subjects)))]
            choices = pool[label][subject]
            selected.append(self.rows[tid][choices[int(rng.integers(len(choices)))]])
        return tid, selected

    def evaluation(self, tid, split, cap=None):
        if split not in {'valid', 'test'}:
            raise ValueError('Evaluation split must be explicit')
        rows = [r for r in self.rows[tid] if r['split'] == split]
        if cap and len(rows) > cap:
            # Fixed label-independent subset; no model/score-dependent selection.
            rng = np.random.default_rng(np.random.SeedSequence([42, tid, 732]))
            rows = [rows[i] for i in sorted(rng.choice(len(rows), cap, replace=False))]
        return rows


@lru_cache(maxsize=24)
def mmap_array(path):
    return np.load(path, mmap_mode='r')
