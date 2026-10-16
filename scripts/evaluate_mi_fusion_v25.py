"""Evaluate source-selected mixing weight; never sweep target weights."""
import hashlib
import json
import pickle
from pathlib import Path
import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits
from eeg_tasks.mi_fusion_v25 import ROOT, mix
from eeg_tasks.mi_covariance_v23 import BANDS, covariance_vectors
from eeg_tasks.source_probe import score


def main():
    source = json.loads((ROOT / 'source_complete.json').read_text())
    for model in source['frozen'].values():
        assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() == model['sha256']
    out = ROOT / 'external_2b'
    out.mkdir(exist_ok=False)
    alpha = source['selected']['alpha_reve']
    (out / 'frozen.json').write_text(json.dumps({'alpha_reve': alpha, 'models': source['frozen']}, indent=2))
    manifest = Path('artifacts/reve_mi_multi/manifests_v2/external_2b.jsonl')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    previous = Path('artifacts/mi_reve_scaling_v24/external_2b')
    assert digest == json.loads((previous / 'results.json').read_text())['manifest_sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 2840
    filters = [butter(4, band, btype='bandpass', fs=200, output='sos') for band in BANDS]
    features = []
    for i, row in enumerate(rows):
        with open(row['path'], 'rb') as f:
            item = pickle.load(f)
        assert item['freq'] == 200 and ['left_hand', 'right_hand'].index(item['label']) == row['label']
        x = np.stack([item['eeg'][c] for c in ['C3', 'CZ', 'C4']]).astype(np.float64)
        features.append(np.concatenate([covariance_vectors(sosfiltfilt(sos, x, axis=-1)[:, :600])['relative_covariance'] for sos in filters]))
        if i % 500 == 0:
            print('extracted', i, flush=True)
    a = joblib.load(source['frozen']['covariance']['path'])
    b = joblib.load(source['frozen']['reve']['path'])
    assert a.classes_.tolist() == b.classes_.tolist() == [0, 1]
    cov = a.predict_proba(np.asarray(features, np.float32))
    reve = b.predict_proba(np.load(previous / 'common_scale_features.npy').astype(np.float32))
    assert np.array_equal(cov.argmax(1), np.load('artifacts/mi_covariance_v23/external_2b/relative_covariance_predictions.npy'))
    assert np.array_equal(reve.argmax(1), np.load(previous / 'common_scale_predictions.npy'))
    probabilities = {'covariance': cov, 'reve': reve, 'selected_mixture': mix(cov, reve, alpha)}
    y = np.array([r['label'] for r in rows])
    results = {}
    for name, p in probabilities.items():
        results[name] = score(rows, y, p.argmax(1))
        results[name]['trial_accuracy'] = float(np.mean(p.argmax(1) == y))
        np.save(out / f'{name}_probabilities.npy', p)
    delta = []
    for subject in sorted({r['subject'] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r['subject'] == subject]
        delta.append(float(np.mean(probabilities['selected_mixture'][ids].argmax(1) == y[ids]) - np.mean(cov[ids].argmax(1) == y[ids])))
    boot = np.random.default_rng(2026).choice(delta, (10000, len(delta)), replace=True).mean(1)
    (out / 'results.json').write_text(json.dumps({'results': results, 'alpha_reve': alpha,
        'mixture_minus_covariance': float(np.mean(delta)), 'paired_subject_95ci': np.quantile(boot, [.025, .975]).tolist(),
        'manifest_sha256': digest, 'note': 'Exploratory repeated-target evaluation; no target weight sweep or selection.'}, indent=2))
    print({k: v['trial_accuracy'] for k, v in results.items()}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
