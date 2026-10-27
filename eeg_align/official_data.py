"""Pinned official masks, original row identities, and fail-closed test access."""
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from scripts.prepare_official_splits import UPSTREAM_COMMIT, import_file, load_source, sha256, write_json
from .benchmark_data import catalog


def check_indices(train, test, subjects, usage=None):
    subjects = np.asarray(subjects).astype(str)
    for ix in (train, test):
        if ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix):
            raise ValueError('Indices must be nonempty one-dimensional integer arrays')
        if len(np.unique(ix)) != len(ix) or ix.min() < 0 or ix.max() >= len(subjects):
            raise ValueError('Repeated or out-of-range source row')
    if np.intersect1d(train, test).size or set(subjects[train]) & set(subjects[test]):
        raise ValueError('Train/test row or subject overlap')
    if usage is not None:
        usage = np.asarray(usage)
        if not np.all(usage[train] == 'train') or not np.all(usage[test] == 'test'):
            raise ValueError('Original usage violated')


def epoch_batches(indices, seed, epoch, batch_size):
    """Every train row exactly once per epoch, including the final partial batch."""
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(indices)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def require_final_test(step, total_steps, smoke=False):
    if smoke or step != total_steps:
        raise RuntimeError('Test access is forbidden before the fixed final update')


def build_plan(config):
    root = Path(config['split_root'])
    summary = json.loads((root / 'complete.json').read_text())
    if summary['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('Wrong benchmark revision')
    jobs, pending = [], {}
    for name, entry in summary['datasets'].items():
        if entry['status'] != 'prepared':
            pending[name] = entry
            continue
        path = root / name / 'manifest.json'
        if sha256(path) != entry['manifest_sha256']:
            raise ValueError(f'Manifest changed: {name}')
        manifest = json.loads(path.read_text())
        for fold in manifest['folds']:
            if fold['seed'] not in config['seeds']:
                continue
            n = fold['train']['n']
            jobs.append(dict(dataset=name, seed=fold['seed'], fold=fold['fold'],
                train_n=n, test_n=fold['test']['n'], train_subjects=fold['train']['subjects'],
                test_subjects=fold['test']['subjects'],
                updates=math.ceil(n / config['batch_size']) * config['epochs'],
                exposures=n * config['epochs'],
                indices_file=fold['indices_file'], indices_sha256=fold['indices_sha256']))
    return dict(protocol=config['protocol'], jobs=jobs, pending=pending,
                split_summary_sha256=sha256(root / 'complete.json'),
                planned_runs=len(jobs), planned_updates=sum(j['updates'] for j in jobs),
                training_started=False, all_13_datasets_ready=not pending)


class OfficialFold:
    def __init__(self, config, dataset, seed, fold):
        root = Path(config['split_root'])
        bench = Path(config['benchmark_root'])
        report = json.loads((root / 'complete.json').read_text())
        actual_commit = subprocess.check_output(['git', '-C', str(bench), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_commit != UPSTREAM_COMMIT or sha256(bench / 'utils/dataset_split.py') != report['split_code_sha256']:
            raise ValueError('Official splitter provenance changed')
        self.manifest = json.loads((root / dataset / 'manifest.json').read_text())
        if sha256(root / dataset / 'manifest.json') != report['datasets'][dataset]['manifest_sha256']:
            raise ValueError('Manifest hash mismatch')
        self.spec = next(x for x in self.manifest['folds'] if x['seed'] == seed and x['fold'] == fold)
        indices = root / self.spec['indices_file']
        if sha256(indices) != self.spec['indices_sha256']:
            raise ValueError('Locked split indices changed')
        with np.load(indices) as masks:
            self.train = masks['train'].copy()
            self.test = masks['test'].copy()
        source = Path(self.manifest['source'])
        if sha256(source) != self.manifest['source_sha256']:
            raise ValueError('Original PKL changed after split lock')
        schema = import_file('official_schema_training', bench / 'utils/EEGDataLoader.py')
        self.data = load_source(source, schema)
        check_indices(self.train, self.test, self.data.subject_ids, getattr(self.data, 'usage', None))
        self.task = next(i for i, t in enumerate(catalog()) if t['task_id'] == dataset)
        self.dataset = dataset
        self.labels = np.asarray(self.data.labels).reshape(-1)
        if not np.all(np.isfinite(self.labels)) or not np.all(self.labels == self.labels.astype(int)):
            raise ValueError('Invalid integer class labels')
        self.labels = self.labels.astype(int)
        if self.labels.min() < 0 or self.labels.max() >= len(catalog()[self.task]['classes']):
            raise ValueError('Source label outside fixed catalog')
        self.subjects = np.asarray(self.data.subject_ids).astype(str)
        self.train_members = set(self.train.tolist())
        self.test_members = set(self.test.tolist())
        for split in ('train', 'test'):
            ids = getattr(self, split)
            if len(ids) != self.spec[split]['n'] or dict(Counter(map(str, self.labels[ids]))) != self.spec[split]['labels']:
                raise ValueError(f'Row count/label audit mismatch: {split}')

    def rows(self, indices, split):
        allowed = self.train_members if split == 'train' else self.test_members if split == 'test' else None
        if allowed is None or not set(map(int, indices)) <= allowed:
            raise ValueError('Attempt to access rows outside declared split')
        return [dict(id=f'{self.dataset}-{i}', parent_id=f'{self.dataset}-{i}',
            dataset=self.dataset, task=self.task, source_index=int(i),
            label=int(self.labels[i]), subject=str(self.subjects[i]), split=split) for i in indices]
