"""Evaluate both source-selected models on the same complete SEED-VII trials."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from eeg_tasks.emotion_trial_v29 import ROOT, trial_groups
from eeg_tasks.emotion_transfer_probe import readrows, metrics


def main():
    source = json.loads((ROOT / 'complete.json').read_text())
    frozen = {}
    for arm, record in source['arms'].items():
        frozen[arm] = hashlib.sha256((ROOT / f'{arm}.joblib').read_bytes()).hexdigest()
        assert frozen[arm] == record['sha256']
    assert hashlib.sha256((ROOT / 'scaler.joblib').read_bytes()).hexdigest() == source['scaler_sha256']
    out = ROOT / 'external_seed_vii';out.mkdir(exist_ok=False)
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    cache = Path('artifacts/emotion_power_v26/external_seed_vii')
    previous = Path('artifacts/emotion_alignment_v21/external_seed_vii')
    rows, arrays = [], []
    for shard in range(4):
        done = json.loads((cache / f'{shard}.done.json').read_text())
        path = previous / f'rows_{shard}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == done['rows_sha256']
        assert hashlib.sha256((cache / f'{shard}.npz').read_bytes()).hexdigest() == done['feature_sha256']
        rows.extend(dict(r, split='external') for r in readrows(path))
        with np.load(cache / f'{shard}.npz') as d:arrays.append(d['per_channel'])
    x = np.concatenate(arrays);groups = trial_groups(rows)
    assert len(rows) == 6400 and len(groups) == 640 and all(len(g) == 10 for g in groups)
    trialrows = [dict(rows[g[0]], windows=len(g)) for g in groups]
    means = np.stack([x[g].mean(0) for g in groups])
    z = joblib.load(ROOT / 'scaler.joblib').transform(means)
    results = {}
    for arm in source['arms']:
        pred = joblib.load(ROOT / f'{arm}.joblib').predict(z)
        results[arm] = metrics(trialrows, pred)
        np.save(out / f'{arm}_predictions.npy', pred)
    a, b = [results[k]['per_subject'] for k in ['window_loss', 'trial_loss']]
    delta = np.array([b[s]['balanced_accuracy'] - a[s]['balanced_accuracy'] for s in sorted(a)])
    boot = np.random.default_rng(2026).choice(delta, (10000, len(delta)), replace=True).mean(1)
    (out / 'results.json').write_text(json.dumps({'results': results, 'trial_minus_window_loss': float(delta.mean()),
        'paired_subject_95ci': np.quantile(boot, [.025, .975]).tolist(), 'frozen': frozen,
        'note': 'Trial-context exploratory evaluation, not 3s online performance. No target fitting/selection.'}, indent=2))
    print({k: v['subject_balanced_accuracy'] for k, v in results.items()})


if __name__ == '__main__':
    with threadpool_limits(limits=4):main()
