"""The historical parent cache, with exact montage/resampling/chunk semantics."""
import csv
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
import numpy as np
import torch
from scipy.signal import resample_poly
from .common import CONFIG, PARENT, PARENT_CONFIG, ROOT, digest


class SignalBatch:
    """CPU signal chunks with trial-indexing semantics, never cached embeddings."""
    def __init__(self, pieces):
        self.pieces = pieces
        if not pieces or len({len(p) for p in pieces}) != 1:
            raise ValueError('Inconsistent signal chunk batches')

    def __len__(self):
        return len(self.pieces[0])

    def __getitem__(self, indices):
        return SignalBatch([p[indices] for p in self.pieces])

    @classmethod
    def prepare(cls, store, rows):
        chunks = None
        adapter = PreparedAdapter(store)
        for offset in range(0, len(rows), CONFIG['encode_batch']):
            part = rows[offset:offset + CONFIG['encode_batch']]
            groups = adapter.batch(part)
            if len(groups) != 1 or groups[0][1] != list(range(len(part))):
                raise ValueError('Personal pilot requires uniform ordered signal batches')
            pieces = groups[0][2]
            if chunks is None:
                chunks = [[] for _ in pieces]
            if len(chunks) != len(pieces):
                raise ValueError('Personal signal lengths changed')
            for destination, piece in zip(chunks, pieces):
                destination.append(torch.from_numpy(piece))
        return cls([torch.cat(parts) for parts in chunks])


def trial_key(row):
    if row['dataset'] == 'bcic_2a':
        return row['record'], int(row['row']) // 48, int(row['event'])
    return row['record'], int(row['event']), int(row['start'])


def source_group(row):
    if row['dataset'] == 'bcic_2a':
        return trial_key(row)
    return row['record'], int(row['event'])


class TrialStore:
    def __init__(self):
        metadata = json.loads((PARENT / 'config.json').read_text())
        catalog = json.loads((ROOT / CONFIG['catalog']).read_text())
        if catalog['source_catalog_sha256'] != metadata['code_sha256']['configs/benchmark_tasks.json']:
            raise RuntimeError('Task catalog does not originate from the shared parent')
        self.tasks = [t for t in catalog['tasks'] if t['task_id'] in CONFIG['datasets']]
        if [t['task_id'] for t in self.tasks] != CONFIG['datasets']:
            raise RuntimeError('Catalog/dataset order changed')
        self.rows, self.meta, self.arrays, self.stats, self.train_subjects = {}, {}, {}, {}, {}
        self.inventory = {}
        for tid, task in enumerate(self.tasks):
            name = task['task_id']
            folder = ROOT / CONFIG['cache_root'] / name
            expected = metadata['cache_indexes'][name]
            if digest(folder / 'index.csv') != expected:
                raise RuntimeError('Historical parent/cache index mismatch')
            complete = json.loads((folder / 'complete.json').read_text())
            manifest = json.loads((ROOT / CONFIG['split_root'] / name / 'manifest.json').read_text())
            if complete['audit_only'] or complete['index_sha256'] != expected or complete['records_sha256'] != manifest['records_sha256']:
                raise RuntimeError('Unverified cache allocation')
            if digest(ROOT / CONFIG['split_root'] / name / 'records.csv') != manifest['records_sha256']:
                raise RuntimeError('Source record allocation changed')
            self.rows[tid] = []
            self.meta[tid] = dict(index_sha256=expected)
            self.train_subjects[tid] = set()
            counts, subjects = Counter(), defaultdict(set)
            with (folder / 'index.csv').open() as handle:
                for i, row in enumerate(csv.DictReader(handle)):
                    counts[row['split']] += 1
                    subjects[row['split']].add(row['subject'])
                    if row['split'] == 'train':
                        self.train_subjects[tid].add(row['subject'])
                    allowed = CONFIG['cohorts'][name]['subjects']
                    if row['subject'] not in allowed:
                        continue
                    row.update(id=f'{name}:{i}', task=tid, dataset=name, label=int(row['label']),
                               samples=int(row['samples']), hz=float(row['hz']), record=row['record_id'])
                    self.rows[tid].append(row)
                    path = folder / row['array']
                    if str(path) not in self.stats:
                        st = path.stat()
                        self.stats[str(path)] = (st.st_size, st.st_mtime_ns)
            self.inventory[name] = dict(window_counts=dict(counts),
                subject_counts={k:len(v) for k,v in subjects.items()},
                subjects_total=len(set.union(*subjects.values())), windows_total=sum(counts.values()))

    def signal(self, row):
        path = ROOT / CONFIG['cache_root'] / row['dataset'] / row['array']
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != self.stats[str(path)]:
            raise RuntimeError('Historical input array changed')
        if str(path) not in self.arrays:
            self.arrays[str(path)] = np.load(path, mmap_mode='r')
        x = np.asarray(self.arrays[str(path)][int(row['row'])], dtype=np.float32)
        if x.shape != (len(json.loads(row['channels'])), row['samples']) or not np.isfinite(x).all():
            raise RuntimeError('Invalid cached EEG')
        return x


def bipolar_matrix(channels):
    pairs = [name.split('-') for name in channels]
    if not all(len(pair) == 2 for pair in pairs):
        raise ValueError('Mixed or invalid bipolar montage')
    electrodes = list(dict.fromkeys(e for pair in pairs for e in pair))
    incidence = np.zeros((len(pairs), len(electrodes)))
    for i, (positive, negative) in enumerate(pairs):
        incidence[i, electrodes.index(positive)] = 1
        incidence[i, electrodes.index(negative)] = -1
    return np.linalg.pinv(incidence).astype(np.float32), electrodes, incidence


def to_electrodes(x, channels):
    x = np.asarray(x, dtype=np.float32)
    if any('-' in ch for ch in channels):
        _, electrodes, incidence = bipolar_matrix(channels)
        # Benchmark's EDF loader fills missing leads with zeros. They are missing
        # measurements, not constraints that the two electrodes have equal voltage.
        active = np.any(x != 0, axis=-1)
        recovered = np.zeros((len(x), len(electrodes), x.shape[-1]), dtype=np.float32)
        residual = 0.0
        for pattern in np.unique(active, axis=0):
            if not pattern.any():
                raise ValueError('All-zero EEG epoch')
            ix = np.flatnonzero(np.all(active == pattern, axis=1))
            observed = x[ix][:, pattern]
            design = incidence[pattern]
            values = np.einsum('ec,bct->bet', np.linalg.pinv(design), observed)
            reconstructed = np.einsum('ce,bet->bct', design, values)
            error = float(np.linalg.norm(reconstructed-observed) / max(np.linalg.norm(observed), 1e-12))
            residual = max(residual, error)
            if error > .05:
                raise ValueError(f'Bipolar montage inconsistent: relative residual={error}')
            recovered[ix] = values
        return recovered, electrodes, {'montage': 'minimum_norm_bipolar', 'relative_residual': residual,
                                        'missing_lead_instances': int((~active).sum()),
                                        'limitation': 'Component references unidentifiable; not recovered monopolar EEG'}
    return x, list(channels), {'montage': 'original_electrodes'}


def resample(x, source_hz):
    ratio = Fraction(200 / float(source_hz)).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)


def segment(x, maximum=2000):
    """Preserve complete epochs using ordered <=10s chunks, never truncate a tail."""
    if x.shape[-1] < 100:
        raise ValueError('LEAF requires at least 0.5 seconds at 200 Hz')
    count = int(np.ceil(x.shape[-1] / maximum))
    # Avoid a tiny final chunk: split approximately equally instead.
    return np.array_split(x, count, axis=-1)



def normalize_trials(x):
    x = np.asarray(x, dtype=np.float32)
    limits = np.percentile(x, [.1, 25, 50, 75, 99.9], axis=(1, 2))
    scale = limits[3] - limits[1]
    fallback = scale <= 0
    scale[fallback] = x[fallback].std(axis=(1, 2))
    if (scale <= 0).any() or not np.isfinite(x).all():
        raise ValueError('Nonfinite or constant EEG')
    return (np.clip(x, limits[0, :, None, None], limits[4, :, None, None])
            - limits[2, :, None, None]) / scale[:, None, None]


class PreparedAdapter:
    def __init__(self, store):
        self.store, self.maps = store, {}
        sys.path.insert(0, PARENT_CONFIG['leaf_root'])
        from leaf_datasets.electrode_unifier import ElectrodeUnifier
        self.unifier = ElectrodeUnifier('LEAF-ch65')
        self.aliases = {k.upper(): k for k in self.unifier.reverse_lookup}

    def spatial(self, channels):
        key = tuple(channels)
        if key not in self.maps:
            names = bipolar_matrix(channels)[1] if any('-' in c for c in channels) else channels
            keep = [i for i,n in enumerate(names) if n.upper() not in {'A1','A2','M1','M2'}]
            aliases = {'T1':'FT9', 'T2':'FT10'}
            canonical = [self.aliases[aliases.get(names[i].upper(),names[i].upper())] for i in keep]
            matrix = self.unifier.map_to_template(np.eye(len(keep),dtype=np.float32)[None],canonical)[0]
            self.maps[key] = keep, matrix
        return self.maps[key]

    def batch(self, rows):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[(row['task'],row['channels'],row['hz'],row['samples'])].append(i)
        result = []
        for (tid, channel_json, hz, _), indices in groups.items():
            channels = json.loads(channel_json)
            keep, spatial = self.spatial(channels)
            raw = np.stack([self.store.signal(rows[i]) for i in indices])
            x, _, _ = to_electrodes(raw,channels)
            x = resample(x[:,keep],hz) if hz != 200 else x[:,keep]
            x = normalize_trials(x)
            x = np.einsum('ec,bct->bet',spatial,x).astype(np.float32)
            chunks = segment(x)
            pieces = [np.ascontiguousarray(np.pad(v,((0,0),(0,0),(0,(-v.shape[-1])%100)))) for v in chunks]
            result.append((tid,indices,pieces))
        return result
