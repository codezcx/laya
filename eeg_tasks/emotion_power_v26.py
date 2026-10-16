"""Preserve or remove spatial amplitude ratios with matched spectral dimensions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.special import logsumexp
from .emotion_transfer_probe import BANDS
from .emotion_reference import prepare

ROOT = Path('artifacts/emotion_power_v26')
MANIFEST = Path('artifacts/faced_scale_v22/rows.jsonl')


def power_features(x, mode):
    x = np.asarray(x, np.float64)
    if x.shape != (30, 600) or not np.isfinite(x).all():
        raise ValueError('Expected finite common30 three-second EEG')
    f, p = welch(x, fs=200, nperseg=200, noverlap=100, axis=-1)
    power = np.stack([p[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1)
    if np.any(power.sum(-1) <= 0):
        raise ValueError('Flat channel')
    logp = np.log(np.maximum(power, np.finfo(np.float64).tiny))
    if mode == 'per_channel':
        return logp - logsumexp(logp, axis=-1, keepdims=True)
    if mode == 'joint_power':
        return logp - logsumexp(logp)
    raise ValueError(mode)


def main(shard):
    assert json.loads(Path('artifacts/faced_full_v22/cohort_status.json').read_text())['state'] == 'complete'
    rows = [json.loads(s) for s in MANIFEST.read_text().splitlines()]
    assert len(rows) == 32827 and {r['split'] for r in rows} == {'train', 'validation'}
    functions = {mode: (lambda x, m=mode: power_features(x, m)) for mode in ['per_channel', 'joint_power']}
    prepare(shard, root=ROOT, manifest_path=MANIFEST, feature_functions=functions)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, choices=range(4), required=True)
    main(p.parse_args().shard)
