"""Frozen native LEAF control, sharing the current sampler, JEV and trainer.

Signal mapping and query extraction match the archived 741fe34 LEAF path.
Only task descriptions condition LEAF; trial labels never enter this module.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

from .runtime import CONFIG, digest
from .benchmark_data import catalog, bipolar_matrix, to_electrodes, resample, segment
from .multitask_data import mmap_array


def upstream_files(config):
    root = Path(config['leaf_root'])
    return [root / p for p in ('LEAF.py', 'load_config.py', 'utils.py',
        'configs/LEAF_mpnet.yaml', 'configs/env.yaml',
        'configs/LEAF-ch65/LEAF-ch65.bvef', 'configs/LEAF-ch65/LEAF-ch65.json',
        'leaf_datasets/electrode_unifier.py')] + [Path(config['text_root']) / p
                                                  for p in ('instructions.json', 'instructions.npy')]


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
        for i, row in enumerate(rows):
            groups[(row['task'], row['channels'], row['hz'], row['samples'])].append(i)
        result, tasks = [], catalog()
        for (tid, channel_json, hz, _), indices in groups.items():
            channels = json.loads(channel_json)
            keep, spatial = self.spatial(channels)
            name = tasks[tid]['task_id']
            folder = Path(CONFIG['cache_root']) / name
            raw = np.stack([mmap_array(str(folder / rows[i]['array']))[int(rows[i]['row'])] for i in indices])
            x, _, info = to_electrodes(raw, channels)
            x = resample(x[:, keep], hz) if hz != 200 else x[:, keep]
            limits = np.percentile(x, [.1, 25, 50, 75, 99.9], axis=(1, 2))
            scale = limits[3] - limits[1]
            fallback = scale <= 0
            scale[fallback] = x[fallback].std(axis=(1, 2))
            if (scale <= 0).any() or not np.isfinite(x).all():
                raise ValueError('Nonfinite or constant LEAF input; no silent row exclusion')
            x = (np.clip(x, limits[0, :, None, None], limits[4, :, None, None]) - limits[2, :, None, None]) / scale[:, None, None]
            x = np.einsum('ec,bct->bet', spatial, x).astype(np.float32)
            if name in {'seed', 'seed_vii'}:
                x = np.clip(x, -10, 10)
            pieces = [np.pad(v, ((0,0), (0,0), (0, (-v.shape[-1]) % 100))) for v in segment(x)]
            self.audit[name] = dict(montage=info['montage'], input_channels=channels,
                output_channels=65, normalization='per-window robust IQR; no fitted split statistics',
                segment_samples=[p.shape[-1] for p in pieces])
            result.append((tid, indices, pieces))
        return result


class FrozenLEAF(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        sys.path.insert(0, config['leaf_root'])
        from LEAF import LEAF
        from load_config import build_model_config, load_yaml
        if digest(config['leaf_checkpoint']) != config['leaf_checkpoint_sha256']:
            raise ValueError('LEAF release checkpoint changed')
        self.leaf = LEAF(build_model_config(load_yaml(Path(config['leaf_root']) / 'configs/LEAF_mpnet.yaml')))
        self.leaf.load_state_dict(torch.load(config['leaf_checkpoint'], map_location='cpu', weights_only=True), strict=True)
        root = Path(config['text_root'])
        info = json.loads((root / 'instructions.json').read_text())
        if digest(root / 'instructions.npy') != info['vectors_sha256']:
            raise ValueError('LEAF task-description embeddings changed')
        values = np.load(root / 'instructions.npy', allow_pickle=False)
        self.register_buffer('instructions', torch.from_numpy(np.stack(
            [values[info['texts'].index(t['instruction'])] for t in catalog()])))
        self.width = self.leaf.qformer.final_norm.normalized_shape[0]
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Freeze both parameters AND running statistics/dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch, count):
        outputs = [None] * count
        for tid, indices, pieces in batch:
            instruction = self.instructions[tid:tid+1].expand(len(indices), -1)
            queries = []
            for piece in pieces:
                x = torch.from_numpy(piece.copy()).to(self.instructions.device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tokens = self.leaf.tower(x)
                    _, history = self.leaf.qformer(tokens, instruction, return_all_q=True)
                    q = self.leaf.qformer.final_norm(history[-1])
                queries.append(q.float())
            joined = torch.cat(queries, dim=1)
            for i, value in zip(indices, joined):
                outputs[i] = value
        if any(q is None for q in outputs):
            raise ValueError('Missing LEAF sample')
        tokens = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
        lengths = torch.tensor([len(q) for q in outputs], device=tokens.device)
        return tokens, torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]


class NativeLEAFReadout(torch.nn.Module):
    output_tokens = 8  # Per <=10-second segment; concatenated for longer windows.

    def forward(self, tokens, valid):
        return tokens, valid
