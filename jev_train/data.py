"""Read the exact native LEAF HDF5 allocations, without repeating preprocessing."""
from collections import defaultdict, OrderedDict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .runtime import CONFIG, ROOT, digest, write_json


class TrialStore:
    def __init__(self, config=None, tasks=None):
        self.config = CONFIG if config is None else config
        manifest_path = ROOT / self.config['manifest']
        if digest(manifest_path) != self.config['manifest_sha256']:
            raise ValueError('LEAF13 manifest identity changed')
        self.manifest = json.loads(manifest_path.read_text())
        self.tasks = (json.loads((ROOT / 'configs/jev13_tasks.json').read_text())['tasks']
                      if tasks is None else tasks)
        if [t['task_id'] for t in self.tasks] != self.config['datasets']:
            raise ValueError('Task order differs from configured datasets')
        self.rows, self.meta, self.pools, self.excluded = {}, {}, {}, {}
        self.handles = OrderedDict()
        self.shard_stats = {}
        for tid, task in enumerate(self.tasks):
            ds = task['task_id']
            entry = self.manifest['datasets'][ds]
            rows, seen = [], set()
            for split in ('train', 'valid', 'test'):
                before = len(rows)
                for shard in entry['splits'][split]:
                    path = Path(shard['path'])
                    stat = path.stat()
                    self.shard_stats[str(path)] = (stat.st_size, stat.st_mtime_ns)
                    with h5py.File(path, 'r', locking=False) as f:
                        labels = f[shard['group'] + '/Y'][:]
                        shape = f[shard['group'] + '/X'].shape
                    if len(shape) != 3 or shape[1] != 65 or not 100 <= shape[2] <= 2000:
                        raise ValueError(f'Invalid prepared signal shape: {ds}: {shape}')
                    if shape[0] != len(labels):
                        raise ValueError('Signal and label counts differ')
                    for index in shard['indices']:
                        if not 0 <= index < len(labels):
                            raise ValueError('Out-of-range manifest index')
                        label = int(labels[index])
                        if not 0 <= label < len(task['classes']):
                            raise ValueError(f'Out-of-range label: {ds}: {label}')
                        sample_id = f"{ds}:{shard['subject']}:{shard['group']}:{index}"
                        if sample_id in seen:
                            raise ValueError('Sample duplicated across allocations')
                        seen.add(sample_id)
                        rows.append(dict(id=sample_id, task=tid, split=split, label=label,
                            subject=str(shard['subject']), path=str(path), group=shard['group'],
                            index=index, samples=shape[2]))
                if len(rows) - before != entry['counts'][split]['windows']:
                    raise ValueError('Manifest window count mismatch')
            self.rows[tid] = rows
            self.excluded[tid] = set()
            self.meta[tid] = dict(index_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
            pool = defaultdict(lambda: defaultdict(list))
            for i, row in enumerate(rows):
                if row['split'] == 'train':
                    pool[row['label']][row['subject']].append(i)
            if set(pool) != set(range(len(task['classes']))):
                raise ValueError(f'Missing training classes: {ds}')
            self.pools[tid] = pool

    def sample(self, step, batch, seed):
        # Exact task/class/subject sampler from historical TrialStore.sample.
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
            rng = np.random.default_rng(np.random.SeedSequence([42, tid, 732]))
            rows = [rows[i] for i in sorted(rng.choice(len(rows), cap, replace=False))]
        return rows

    def signal(self, row):
        path = row['path']
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != self.shard_stats[path]:
            raise RuntimeError('Input shard changed during this run')
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r', locking=False)
        self.handles.move_to_end(path)
        while len(self.handles) > 16:
            _, handle = self.handles.popitem(last=False)
            handle.close()
        x = np.asarray(self.handles[path][row['group'] + '/X'][row['index']], dtype=np.float32)
        if x.shape != (65, row['samples']) or not np.isfinite(x).all():
            raise ValueError('Invalid prepared EEG sample')
        return x


class PreparedAdapter:
    def __init__(self, store):
        self.store = store
        self.audit = dict(source='datasets/leaf13/manifest.json', preprocessing_reapplied=False,
                          channels=65, fs_hz=200, input_lengths=[], samples_forwarded=0)

    def batch(self, rows):
        lengths = {r['samples'] for r in rows}
        if len(lengths) != 1:
            raise ValueError('Each microbatch must have one native window length')
        length = lengths.pop()
        if length not in self.audit['input_lengths']:
            self.audit['input_lengths'].append(length)
        self.audit['samples_forwarded'] += len(rows)
        signals = np.stack([self.store.signal(r) for r in rows])
        return [(length, list(range(len(rows))), [signals])]


def classification_metrics(labels, probabilities, classes):
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(probabilities)[:, :classes].argmax(-1)
    cm = np.zeros((classes, classes), dtype=int)
    np.add.at(cm, (y, pred), 1)
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.full(classes, np.nan), where=support > 0)
    return dict(n=len(y), accuracy=float(np.mean(y == pred)),
                balanced_accuracy=float(np.nanmean(recall)), supported_classes=int((support > 0).sum()),
                confusion_matrix=cm.tolist())


def preflight():
    store = TrialStore()
    native_tasks = yaml.safe_load((Path(CONFIG['leaf_root']) / 'configs/tasks.yaml').read_text())['datasets']
    result = dict(manifest_sha256=CONFIG['manifest_sha256'], datasets={}, total_windows={s: 0 for s in ('train', 'valid', 'test')})
    for tid, task in enumerate(store.tasks):
        name = task['task_id']
        if [c['native_label'] for c in task['classes']] != native_tasks[name]:
            raise ValueError(f'Native label order changed: {name}')
        rows = store.rows[tid]
        counts = {s: sum(r['split'] == s for r in rows) for s in result['total_windows']}
        for split, n in counts.items():
            result['total_windows'][split] += n
        pool = [r for r in rows if r['split'] == 'train']
        probes = np.linspace(0, len(pool) - 1, min(32, len(pool)), dtype=int)
        for index in probes:
            x = store.signal(pool[index])
            if not np.any(x):
                raise ValueError(f'All-zero training signal: {name}')
        result['datasets'][name] = dict(counts=counts, classes=[c['native_label'] for c in task['classes']],
            index_sha256=store.meta[tid]['index_sha256'], train_probes=len(probes), window_samples=pool[0]['samples'])
    result['passed'] = True
    write_json(ROOT / CONFIG['output_root'] / 'input_preflight.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    preflight()
