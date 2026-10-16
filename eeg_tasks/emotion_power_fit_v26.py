"""Audit paired raw-power features and select only on held-source validation."""
import hashlib
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .emotion_power_v26 import ROOT, MANIFEST
from .emotion_transfer_probe import readrows, metrics
from .mi_covariance_fit_v23 import fit_model, fold_indices

ARMS = ['per_channel', 'joint_power']
DOMAINS = ['SEED', 'FACED-NEMAR']


def main():
    rows = readrows(MANIFEST)
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    arrays = {a: np.empty((len(rows), 30, 5), np.float32) for a in ARMS}
    seen, hashes = [], {}
    for shard in range(4):
        path = ROOT / 'features' / f'{shard}.npz'
        done = json.loads(path.with_suffix('.done.json').read_text())
        assert done['manifest_sha256'] == digest
        with np.load(path) as d:
            ix = d['indices']
            assert len(ix) == done['n']
            seen.extend(ix.tolist())
            for a in ARMS:
                assert d[a].shape == (len(ix), 30, 5) and np.isfinite(d[a]).all()
                arrays[a][ix] = d[a]
        hashes[str(shard)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sorted(seen) == list(range(len(rows))) and len(rows) == 32827
    assert {r['split'] for r in rows} == {'train', 'validation'}
    ti = [i for i, r in enumerate(rows) if r['split'] == 'train']
    vi = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    assert not {(rows[i]['dataset'], rows[i]['subject']) for i in ti} & {(rows[i]['dataset'], rows[i]['subject']) for i in vi}
    old = np.load('artifacts/faced_scale_v22/features.npy')
    delta = np.abs(old - arrays['per_channel'])
    audit = {'n': len(rows), 'manifest_sha256': digest, 'feature_hashes': hashes,
             'control_max_abs_difference': float(delta.max()), 'control_mean_abs_difference': float(delta.mean())}
    if delta.max() > .001:
        # Explicitly inspected precision case; do not relax the general threshold.
        precision_path = ROOT / 'control_precision_audit.json'
        precision = json.loads(precision_path.read_text())
        bad = set(np.flatnonzero(delta.max((1, 2)) > .001).tolist())
        inspected = {r['index'] for r in precision['rows']}
        assert precision['subject'] == 30 and precision['n'] == len(inspected) == 280
        assert len(bad) == 214 and bad <= inspected
        assert all(rows[i]['dataset'] == 'FACED-NEMAR' and rows[i]['subject'] == '30' for i in inspected)
        assert precision['max_legacy_replay_delta'] < 1e-5 and precision['max_float64_normalization_delta'] < 1e-5
        assert precision['clipped_values'] == 0
        audit['precision_exception'] = {'subject': 30, 'affected_windows': len(bad),
            'evidence_sha256': hashlib.sha256(precision_path.read_bytes()).hexdigest(),
            'action': 'Retain all windows; both v26 arms use raw float64 features. Legacy fp32 control differs.'}
    (ROOT / 'feature_audit.json').write_text(json.dumps(audit, indent=2))
    y = np.array([r['class_id'] for r in rows])
    assert set(y) == {0, 1, 2}
    out = ROOT / 'fits'
    out.mkdir(exist_ok=False)
    results = {}
    for a in ARMS:
        x = arrays[a].reshape(len(rows), 150)
        np.save(ROOT / f'{a}.npy', x)
        sweep = []
        for c in [.001, .01, .1, 1., 10.]:
            folds = {}
            for held in DOMAINS:
                train, validation = fold_indices(rows, held)
                model, warnings = fit_model(x, y, rows, train, c, max_iter=10000)
                if warnings:
                    raise RuntimeError((a, c, held, warnings))
                folds[held] = metrics([rows[i] for i in validation], model.predict(x[validation]))
            value = float(np.mean([m['subject_balanced_accuracy'] for m in folds.values()]))
            sweep.append({'C': c, 'held_domain_mean': value, 'folds': folds})
            print(a, c, value, flush=True)
        selected = max(sweep, key=lambda r: r['held_domain_mean'])
        model, warnings = fit_model(x, y, rows, ti, selected['C'], max_iter=10000)
        if warnings:
            raise RuntimeError((a, warnings))
        joblib.dump(model, out / f'{a}.joblib')
        within = {}
        for ds in DOMAINS:
            ix = [i for i in vi if rows[i]['dataset'] == ds]
            within[ds] = metrics([rows[i] for i in ix], model.predict(x[ix]))
        results[a] = {'selected': selected, 'sweep': sweep, 'within_source_validation': within,
                      'sha256': hashlib.sha256((out / f'{a}.joblib').read_bytes()).hexdigest()}
    (out / 'complete.json').write_text(json.dumps({'arms': results, 'feature_audit': audit,
        'target_evaluation': 'not_run', 'selection': 'Mean held-source subject-balanced accuracy; smaller C on exact ties.'}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
