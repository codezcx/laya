"""Historical LEAF signal preprocessing, independent of instruction/label text."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from .runtime import CONFIG
from .multitask_data import mmap_array
from .benchmark_data import catalog, bipolar_matrix, to_electrodes, resample, segment


class LEAFSignalAdapter:
    def __init__(self):
        sys.path.insert(0, CONFIG['leaf_root'])
        from leaf_datasets.electrode_unifier import ElectrodeUnifier
        self.unifier = ElectrodeUnifier('LEAF-ch65')
        self.aliases = {k.upper(): k for k in self.unifier.reverse_lookup}
        self.maps, self.audit = {}, {}

    def spatial(self, channels):
        key = tuple(channels)
        if key not in self.maps:
            names = bipolar_matrix(channels)[1] if any('-' in c for c in channels) else channels
            keep = [i for i, name in enumerate(names) if name.upper() not in {'A1', 'A2', 'M1', 'M2'}]
            aliases = {'T1': 'FT9', 'T2': 'FT10'}
            canonical = [self.aliases[aliases.get(names[i].upper(), names[i].upper())] for i in keep]
            matrix = self.unifier.map_to_template(np.eye(len(keep), dtype=np.float32)[None], canonical)[0]
            self.maps[key] = keep, matrix
        return self.maps[key]

    def batch(self, rows):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[(row['task'], row['channels'], row['hz'], row['samples'])].append(index)
        result = []
        root = Path(CONFIG['cache_root'])
        tasks = catalog()
        for (tid, channel_json, hz, _), indices in groups.items():
            channels = json.loads(channel_json)
            keep, spatial = self.spatial(channels)
            raw = np.stack([mmap_array(str(root / tasks[tid]['task_id'] / rows[i]['array']))[int(rows[i]['row'])] for i in indices])
            try:
                x, _, info = to_electrodes(raw, channels)
            except ValueError as exc:
                raise ValueError(f'{exc}; batch IDs={[rows[i]["id"] for i in indices]}') from exc
            x = resample(x[:, keep], hz) if hz != 200 else x[:, keep]
            limits = np.percentile(x, [.1, 25, 50, 75, 99.9], axis=(1, 2))
            scale = limits[3] - limits[1]
            fallback = scale <= 0
            scale[fallback] = x[fallback].std(axis=(1, 2))
            if (scale <= 0).any() or not np.isfinite(x).all():
                raise ValueError(f'Nonfinite or constant EEG; IDs={[rows[i]["id"] for i in indices]}')
            x = (np.clip(x, limits[0, :, None, None], limits[4, :, None, None]) - limits[2, :, None, None]) / scale[:, None, None]
            x = np.einsum('ec,bct->bet', spatial, x).astype(np.float32)
            if tasks[tid]['task_id'] in {'seed', 'seed_vii'}:
                x = np.clip(x, -10, 10)
            chunks = segment(x)
            pieces = [np.ascontiguousarray(np.pad(v, ((0, 0), (0, 0), (0, (-v.shape[-1]) % 100)))) for v in chunks]
            self.audit[tasks[tid]['task_id']] = dict(input_channels=channels, montage=info,
                output_channels=65, output_hz=200, normalization='per-epoch 0.1/99.9 percentile clip; median/IQR; std fallback',
                samples_at_200hz=x.shape[-1], segment_samples=[v.shape[-1] for v in chunks],
                right_zero_padding=[p.shape[-1]-v.shape[-1] for p,v in zip(pieces,chunks)],
                output_tokens=sum(2 * (p.shape[-1] // 100) for p in pieces))
            result.append((tid, indices, pieces))
        return result
