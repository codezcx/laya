"""Source-only paired MI covariance diagnostic, with no target adaptation."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path('artifacts/mi_covariance_v23')
SOURCE = Path('artifacts/mi_full_sources_v12')
BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
ARMS = ['correlation', 'relative_covariance']


def covariance_vectors(x):
    """Trace-normalized SPD log versus correlation log, both trial-local."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (3, 600) or not np.isfinite(x).all():
        raise ValueError('Expected finite three-channel three-second window')
    x = x - x.mean(-1, keepdims=True)
    cov = x @ x.T / x.shape[-1]
    if np.any(np.diag(cov) <= 0):
        raise ValueError('Flat channel')
    scale = np.sqrt(np.diag(cov))
    matrices = {'correlation': cov / np.outer(scale, scale),
                'relative_covariance': 3 * cov / np.trace(cov)}
    result = {}
    ix = np.triu_indices(3)
    weights = np.where(ix[0] == ix[1], 1., np.sqrt(2.))
    for arm, matrix in matrices.items():
        # Fixed shrinkage; fitted neither on source nor target collections.
        matrix = .99 * matrix + .01 * np.eye(3)
        eigenvalues, vectors = np.linalg.eigh(matrix)
        logmatrix = (vectors * np.log(eigenvalues)) @ vectors.T
        result[arm] = logmatrix[ix] * weights
    return result


def extract(shard):
    ROOT.mkdir(exist_ok=True)
    out = ROOT / f'{shard}.npz'
    if out.exists():
        raise FileExistsError(out)
    audit = json.loads((SOURCE / 'audit.json').read_text())
    rows = [json.loads(s) for s in (SOURCE / 'manifest.jsonl').read_text().splitlines()]
    keep = np.load(SOURCE / 'keep_indices.npy')
    assert len(keep) == audit['kept_n'] == 37432
    assert all(rows[int(i)]['split'] in ['train', 'validation'] for i in keep)
    filters = [butter(4, band, btype='bandpass', fs=200, output='sos') for band in BANDS]
    indices = keep[shard::4]
    arrays = {arm: [] for arm in ARMS}
    for j, index in enumerate(indices):
        row = rows[int(index)]
        with open(row['path'], 'rb') as stream:
            item = pickle.load(stream)
        assert item['freq'] == 200 and ['left_hand', 'right_hand'].index(item['label']) == row['label']
        x = np.stack([item['eeg'][c] for c in ['C3', 'CZ', 'C4']]).astype(np.float64)
        assert x.shape[-1] >= 600 and np.isfinite(x).all()
        bands = [covariance_vectors(sosfiltfilt(sos, x, axis=-1)[:, :600]) for sos in filters]
        for arm in ARMS:
            arrays[arm].append(np.concatenate([b[arm] for b in bands]))
        if j % 1000 == 0:
            print(shard, j, len(indices), flush=True)
    np.savez(out, indices=indices, **{k: np.asarray(v, np.float32) for k, v in arrays.items()})
    (ROOT / f'{shard}.done.json').write_text(json.dumps({
        'n': len(indices), 'bands': BANDS, 'channels': ['C3', 'Cz', 'C4'],
        'manifest_sha256': hashlib.sha256((SOURCE / 'manifest.jsonl').read_bytes()).hexdigest(),
        'feature_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, choices=range(4), required=True)
    extract(p.parse_args().shard)
