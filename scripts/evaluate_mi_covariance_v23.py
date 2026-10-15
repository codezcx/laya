"""Evaluate both frozen MI covariance models on the existing 2b E-trial manifest."""
import hashlib
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits

from eeg_tasks.mi_covariance_v23 import ROOT, ARMS, BANDS, covariance_vectors
from eeg_tasks.source_probe import score


def main():
    fitted = json.loads((ROOT / 'fits/complete.json').read_text())
    frozen = {}
    for arm in ARMS:
        path = ROOT / 'fits' / f'{arm}.joblib'
        frozen[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert frozen[arm] == fitted['arms'][arm]['sha256']
    out = ROOT / 'external_2b'
    out.mkdir(exist_ok=False)
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    manifest = Path('artifacts/reve_mi_multi/manifests_v2/external_2b.jsonl')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 2840 and len({r['path'] for r in rows}) == 2840
    filters = [butter(4, band, btype='bandpass', fs=200, output='sos') for band in BANDS]
    features = {arm: [] for arm in ARMS}
    for i, row in enumerate(rows):
        with open(row['path'], 'rb') as f:
            item = pickle.load(f)
        assert item['freq'] == 200 and ['left_hand', 'right_hand'].index(item['label']) == row['label']
        x = np.stack([item['eeg'][c] for c in ['C3', 'CZ', 'C4']]).astype(np.float64)
        assert x.shape[-1] >= 600 and np.isfinite(x).all()
        bands = [covariance_vectors(sosfiltfilt(sos, x, axis=-1)[:, :600]) for sos in filters]
        for arm in ARMS:
            features[arm].append(np.concatenate([b[arm] for b in bands]))
        if i % 500 == 0:
            print('extracted', i, flush=True)
    y = np.array([r['label'] for r in rows])
    results, predictions = {}, {}
    for arm in ARMS:
        model = joblib.load(ROOT / 'fits' / f'{arm}.joblib')
        pred = model.predict(np.asarray(features[arm], np.float32))
        predictions[arm] = pred
        results[arm] = score(rows, y, pred)
        results[arm]['trial_accuracy'] = float(np.mean(pred == y))
        results[arm]['shuffled'] = score(rows, y, pred[np.random.default_rng(2026).permutation(len(pred))])
        np.save(out / f'{arm}_predictions.npy', pred)
    subjects = sorted({r['subject'] for r in rows})
    delta = []
    for subject in subjects:
        ids = [i for i, r in enumerate(rows) if r['subject'] == subject]
        delta.append(float(np.mean(predictions['relative_covariance'][ids] == y[ids]) -
                           np.mean(predictions['correlation'][ids] == y[ids])))
    boot = np.random.default_rng(2026).choice(delta, (10000, len(delta)), replace=True).mean(1)
    result = {'results': results, 'relative_minus_correlation': float(np.mean(delta)),
              'paired_subject_95ci': np.quantile(boot, [.025, .975]).tolist(),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'frozen': frozen,
              'note': 'Repeated-target exploratory evaluation; no target fitting or selection. Classical features use no pretrained EEG encoder.'}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print({arm: results[arm]['trial_accuracy'] for arm in ARMS}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
