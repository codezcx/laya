"""Historical BCIC cache and the parent's exact per-trial LEAF preprocessing."""
import csv
import json
import sys
import numpy as np
from .common import CONFIG, PARENT, PARENT_CONFIG, ROOT, digest

TASK = dict(task_id='bcic_2a', role='benchmark',
    instruction='This is a motor imagery decoding task',
    question='Which movement is the participant imagining?',
    classes=[dict(class_id=str(i), option=s) for i, s in enumerate((
        'Left hand motor imagery', 'Right hand motor imagery',
        'Both feet motor imagery', 'Tongue motor imagery'))])


def trial_key(row):
    return row['record'], int(row['row']) // 48, int(row['event'])


class TrialStore:
    def __init__(self):
        self.root = ROOT / CONFIG['cache_root']
        index = self.root / 'index.csv'
        expected = json.loads((PARENT / 'config.json').read_text())['cache_indexes']['bcic_2a']
        if digest(index) != expected:
            raise RuntimeError('Historical parent/cache index mismatch')
        split = json.loads((ROOT / CONFIG['split_manifest']).read_text())['counts']
        if split['train']['subjects'] != ['1', '2', '3', '4', '5']:
            raise RuntimeError('Shared training subject allocation changed')
        for subject, allocation in CONFIG['subjects'].items():
            if subject in split['train']['subjects'] or subject not in split[allocation]['subjects']:
                raise RuntimeError('Target subject is not held out from shared training')
        self.tasks, self.rows = [TASK], {0: []}
        self.meta = {0: dict(index_sha256=expected)}
        self.arrays, self.stats = {}, {}
        with index.open() as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                row.update(id=f'bcic_2a:{i}', task=0, label=int(row['label']),
                           samples=int(row['samples']), hz=int(row['hz']), record=row['record_id'])
                if row['subject'] not in CONFIG['subjects']:
                    continue
                if row['split'] != CONFIG['subjects'][row['subject']] or row['samples'] != 800 or row['hz'] != 200:
                    raise RuntimeError('Historical row contract changed')
                self.rows[0].append(row)
                path = self.root / row['array']
                st = path.stat()
                self.stats[str(path)] = (st.st_size, st.st_mtime_ns)

    def signal(self, row):
        path = self.root / row['array']
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != self.stats[str(path)]:
            raise RuntimeError('Historical input array changed')
        if str(path) not in self.arrays:
            self.arrays[str(path)] = np.load(path, mmap_mode='r')
        x = np.asarray(self.arrays[str(path)][int(row['row'])], dtype=np.float32)
        if x.shape != (22, 800) or not np.isfinite(x).all():
            raise RuntimeError('Invalid BCIC signal')
        return x


def normalize_trials(x):
    """Old 0.1/99.9 clip and median/IQR scaling, before channel interpolation."""
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

    def batch(self, rows):
        channels = json.loads(rows[0]['channels'])
        if any(json.loads(r['channels']) != channels for r in rows):
            raise ValueError('Mixed montage in BCIC batch')
        key = tuple(channels)
        if key not in self.maps:
            canonical = [self.aliases[c.upper()] for c in channels]
            self.maps[key] = self.unifier.map_to_template(np.eye(22, dtype=np.float32)[None], canonical)[0]
        x = normalize_trials(np.stack([self.store.signal(r) for r in rows]))
        x = np.einsum('ec,bct->bet', self.maps[key], x).astype(np.float32)
        return [(0, list(range(len(rows))), [np.ascontiguousarray(x)])]
