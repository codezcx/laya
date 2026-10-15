"""Evaluate every source-frozen scale control without target fitting or selection."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from eeg_tasks.emotion_transfer_probe import CHANNELS, metrics, readrows, transform

ROOT = Path('artifacts/faced_scale_v22')
CACHE = Path('artifacts/emotion_alignment_v21/external_seed_vii')


def main():
    frozen = json.loads((ROOT / 'source_freeze.json').read_text())
    names = [f'{scale}_{arm}' for scale in ['pilot', 'full']
             for arm in ['full_linear', 'spatial_linear']]
    assert set(frozen['models']) == set(names)
    for name in names:
        p = ROOT / name
        complete = json.loads((p / 'complete.json').read_text())
        assert hashlib.sha256((p / 'best.joblib').read_bytes()).hexdigest() == complete['sha256'] == frozen['models'][name]
        assert complete['validation_rows_sha256'] == frozen['validation_rows_sha256']
    out = ROOT / 'external_seed_vii'
    out.mkdir(exist_ok=False)
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    rows, arrays, hashes = [], [], {}
    for shard in range(4):
        assert json.loads((CACHE / f'{shard}.done.json').read_text())['channels'] == CHANNELS
        rr = readrows(CACHE / f'rows_{shard}.jsonl')
        x = np.load(CACHE / f'spectral_{shard}.npy')
        assert x.shape == (len(rr), 150) and np.isfinite(x).all()
        rows.extend(rr)
        arrays.append(x.reshape(-1, 30, 5))
        for filename in [f'rows_{shard}.jsonl', f'spectral_{shard}.npy']:
            hashes[filename] = hashlib.sha256((CACHE / filename).read_bytes()).hexdigest()
    assert len(rows) == 6400 and len({r['subject'] for r in rows}) == 20
    assert {r['dataset'] for r in rows} == {'SEED-VII-HNS'}
    assert len({(r['subject'], r['trial'], r['start_sample']) for r in rows}) == 6400
    raw = np.concatenate(arrays)
    results = {}
    for name in names:
        arm = name.split('_', 1)[1]
        model = joblib.load(ROOT / name / 'best.joblib')
        pred = model.predict(transform(raw, arm))
        results[name] = metrics(rows, pred)
        results[name]['shuffled'] = metrics(rows, pred[np.random.default_rng(2026).permutation(len(pred))])
        np.save(out / f'{name}_predictions.npy', pred)
        print(name, results[name]['subject_balanced_accuracy'], flush=True)
    paired = {}
    for arm in ['full_linear', 'spatial_linear']:
        a, b = results[f'full_{arm}']['per_subject'], results[f'pilot_{arm}']['per_subject']
        assert set(a) == set(b)
        delta = np.array([a[s]['balanced_accuracy'] - b[s]['balanced_accuracy'] for s in sorted(a)])
        boot = np.random.default_rng(2026).choice(delta, (10000, len(delta)), replace=True).mean(1)
        paired[arm] = {'full_minus_pilot': float(delta.mean()),
                       'paired_subject_95ci': np.quantile(boot, [.025, .975]).tolist()}
    (out / 'results.json').write_text(json.dumps({
        'results': results, 'paired': paired, 'cache_sha256': hashes,
        'note': 'Repeated-target exploratory SEED-VII HNS evaluation. All source-frozen arms reported; no target fitting or selection.'
    }, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
