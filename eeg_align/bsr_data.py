"""Unit-explicit BSR adaptation; locked windows and QC remain untouched."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .benchmark_data import catalog, to_electrodes, resample, segment
from .multitask_data import mmap_array
from .runtime import CONFIG

ALIASES = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
# Explicitly unsupported reference/inferior electrodes; logged in audit output.
OUTSIDE_GRID = {'A1', 'A2', 'M1', 'M2', 'T1', 'T2', 'T9', 'T10', 'IZ'}


def pad_for_stft(piece):
    """Smallest right reflect pad admitting exactly 16 centered STFT frames.

    E.g. 161-sample P300 epochs need 4 samples; no observed sample is removed.
    """
    length = piece.shape[-1]
    if length <= 100:
        raise ValueError('Epoch too short for released reflect STFT')
    for extra in range(16):
        n = length + extra
        if any(n // hop == 15 for hop in range(max(1, n // 16), n // 15 + 3)):
            return np.pad(piece, ((0,0),(0,0),(0,extra)), mode='reflect') if extra else piece
    raise ValueError('Unable to construct a 16-frame STFT grid')


class BSRSignalAdapter:
    def __init__(self):
        sys.path.insert(0, str(Path(CONFIG['bsr_root']) / 'code'))
        from bsri.model import CHANNELS_1010
        self.grid = set(CHANNELS_1010)
        self.tasks = catalog()
        self.audit = {}

    def batch(self, rows):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[(row['task'], row['channels'], row['hz'], row['samples'])].append(i)
        result = []
        for (tid, channel_json, hz, _), indices in groups.items():
            name = self.tasks[tid]['task_id']
            channels = [c.upper() for c in json.loads(channel_json)]
            root = Path(CONFIG['cache_root']) / name
            raw = np.stack([mmap_array(str(root / rows[i]['array']))[int(rows[i]['row'])] for i in indices])
            unit = CONFIG['input_units'][name]
            if unit not in {'V', 'uV'}:
                raise ValueError(f'Unresolved input unit: {name}')
            raw = raw * (1e6 if unit == 'V' else 1.)
            x, electrodes, info = to_electrodes(raw, channels)
            canonical = [ALIASES.get(c, c) for c in electrodes]
            dropped = [c for c in canonical if c not in self.grid]
            if not set(dropped) <= OUTSIDE_GRID:
                raise ValueError(f'Unaudited out-of-grid channels: {dropped}')
            keep = [i for i, c in enumerate(canonical) if c in self.grid]
            names = [canonical[i] for i in keep]
            if not keep or len(set(names)) != len(names):
                raise ValueError('Empty or duplicate BSR montage')
            x = x[:, keep]
            if hz != 200:
                x = resample(x, hz)
            if not np.isfinite(x).all() or (x.std(axis=(1, 2)) <= 0).any():
                raise ValueError(f'Invalid BSR input: {[rows[i]["id"] for i in indices]}')
            original_pieces = segment(x)
            pieces = [np.ascontiguousarray(pad_for_stft(p), dtype=np.float32) for p in original_pieces]
            # Shortest current epoch is 0.8s; reflect-STFT needs >100 samples.
            if any(p.shape[-1] <= 100 for p in pieces):
                raise ValueError('Epoch too short for released 200-point reflect STFT')
            self.audit[name] = dict(source_unit=unit, output_unit='uV', input_channels=channels,
                renderer_channels=names, explicitly_dropped=dropped,
                montage=info['montage'], samples_at_200hz=x.shape[-1],
                segment_samples=[p.shape[-1] for p in original_pieces],
                right_reflect_padding=[p.shape[-1]-o.shape[-1] for p,o in zip(pieces,original_pieces)])
            result.append((indices, names, pieces))
        return result


def encode_batch(encoder, adapted, size):
    """Preserve every chunk and mark chunk order; never average away a tail."""
    samples = [None] * size
    for indices, channels, pieces in adapted:
        encoded = []
        for j, piece in enumerate(pieces):
            z = encoder(torch.from_numpy(piece), channels)
            # Relative segment position; zero for a single segment. No text.
            if len(pieces) > 1:
                dims = torch.arange(0, z.shape[-1], 2, device=z.device).float()
                angle = j / torch.pow(10000., dims / z.shape[-1])
                position = torch.stack((angle.sin(), angle.cos() - 1), -1).flatten()
                z = z + position.to(z.dtype)[None, None]
            encoded.append(z)
        joined = torch.cat(encoded, dim=1)
        for i, value in zip(indices, joined):
            samples[i] = value
    if any(s is None for s in samples):
        raise ValueError('Missing encoded row')
    tokens = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
    lengths = torch.tensor([len(s) for s in samples], device=tokens.device)
    valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
    return tokens, valid
