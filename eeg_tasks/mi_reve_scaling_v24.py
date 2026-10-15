"""Paired frozen REVE features; change only per-channel versus common scaling."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from .data import normalized_window, preprocess

ROOT = Path('artifacts/mi_reve_scaling_v24')
SOURCE = Path('artifacts/mi_full_sources_v12')
ARMS = ['per_channel', 'common_scale']


def scale_window(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (3, 600) or not np.isfinite(x).all():
        raise ValueError('Expected finite three-channel three-second window')
    if mode == 'per_channel':
        return normalized_window(x)
    if mode != 'common_scale':
        raise ValueError(mode)
    x = x - x.mean(-1, keepdims=True)
    if np.any(x.std(-1) < 1e-12):
        raise ValueError('Flat EEG channel')
    return np.clip(x / np.sqrt(np.mean(x * x)), -15, 15).astype(np.float32)


def extract(shard):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=True)
    output = ROOT / f'{shard}.npz'
    if output.exists():
        raise FileExistsError(output)
    base = json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    manifest = SOURCE / 'manifest.jsonl'
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    keep = np.load(SOURCE / 'keep_indices.npy')
    assert len(keep) == json.loads((SOURCE / 'audit.json').read_text())['kept_n'] == 37432
    assert all(rows[int(i)]['split'] in ('train', 'validation') for i in keep)
    indices = keep[shard::4]
    old = np.load(SOURCE / f'{shard}.npz')
    old_index = {int(i): j for j, i in enumerate(old['indices'])}
    model = AutoModel.from_pretrained(base['reve_checkpoint'], local_files_only=True,
                                     trust_remote_code=True).cuda().eval().requires_grad_(False)
    bank = AutoModel.from_pretrained(base['positions_checkpoint'], local_files_only=True, trust_remote_code=True)
    pos = bank(['C3', 'Cz', 'C4']).detach().cuda()
    features = {arm: [] for arm in ARMS}
    replay_max = 0.
    clipping = {arm: 0 for arm in ARMS}
    for begin in range(0, len(indices), 64):
        ids = indices[begin:begin + 64]
        signals = {arm: [] for arm in ARMS}
        for index in ids:
            row = rows[int(index)]
            with open(row['path'], 'rb') as f:
                item = pickle.load(f)
            assert item['freq'] == 200 and base['labels'].index(item['label']) == row['label']
            raw = np.stack([item['eeg'][c] for c in ['C3', 'CZ', 'C4']])
            window = preprocess(raw, 200, {'bandpass': [1, 30]})[:, :600]
            for arm in ARMS:
                x = scale_window(window, arm)
                signals[arm].append(x)
                clipping[arm] += int(np.sum(np.abs(x) == 15))
            assert hashlib.sha256(signals['per_channel'][-1].tobytes()).hexdigest() == row['signal_sha256']
        for arm in ARMS:
            x = torch.from_numpy(np.stack(signals[arm])).cuda()
            stride = model.config.patch_size - model.config.patch_overlap
            tail = (-(600 - model.config.patch_size)) % stride
            x = torch.nn.functional.pad(x, (0, tail), mode='replicate')
            with torch.no_grad():
                z = model(x, pos[None].expand(len(x), -1, -1)).mean(2).cpu().numpy().astype(np.float16)
            assert z.shape == (len(ids), 3, 512) and np.isfinite(z).all()
            features[arm].append(z.reshape(len(ids), 1536))
            if arm == 'per_channel':
                previous = old['features'][[old_index[int(i)] for i in ids]]
                error = float(np.max(np.abs(z.astype(np.float32) - previous.astype(np.float32))))
                replay_max = max(replay_max, error)
                if error > .02:
                    raise ValueError(f'Unexpected frozen encoder replay difference: {error}')
        if begin % 1024 == 0:
            print(shard, begin, len(indices), 'replay_max', replay_max, flush=True)
    np.savez(output, indices=indices, **{k: np.concatenate(v) for k, v in features.items()})
    (ROOT / f'{shard}.done.json').write_text(json.dumps({
        'n': len(indices), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'feature_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'replay_max_abs_fp16': replay_max, 'clipped_values': clipping,
        'checkpoint': base['reve_checkpoint'], 'channels': ['C3', 'Cz', 'C4'],
        'bandpass': [1, 30], 'samples': 600, 'dtype': 'fp32 encoder, fp16 cache',
    }, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, choices=range(4), required=True)
    extract(p.parse_args().shard)
