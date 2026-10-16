"""Select covariance/REVE probability mixing exclusively on held-source predictions."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .mi_covariance_fit_v23 import load_cache, fold_indices, fit_model, DOMAINS
from .mi_covariance_v23 import ROOT as COV_ROOT, SOURCE
from .mi_reve_scaling_v24 import ROOT as REVE_ROOT
from .source_probe import score

ROOT = Path('artifacts/mi_fusion_v25')


def mix(cov, reve, alpha):
    if cov.shape != reve.shape or cov.ndim != 2 or cov.shape[1] != 2:
        raise ValueError('Expected matching binary probabilities')
    if not 0 <= alpha <= 1:
        raise ValueError('Mixing weight outside [0,1]')
    return (1 - alpha) * cov + alpha * reve


def main():
    ROOT.mkdir(exist_ok=False)
    rows, cov, cov_provenance = load_cache(COV_ROOT, SOURCE, ['relative_covariance'], 30)
    rr, reve, reve_provenance = load_cache(REVE_ROOT, SOURCE, ['common_scale'], 1536)
    assert rows == rr
    plans = {
        'covariance': (COV_ROOT / 'fits', 'relative_covariance', cov['relative_covariance']),
        'reve': (REVE_ROOT / 'fits_converged', 'common_scale', reve['common_scale']),
    }
    y = np.array([r['label'] for r in rows])
    vi = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    valid_rows = [rows[i] for i in vi]
    probabilities, frozen, folds = {}, {}, {}
    for name, (folder, arm, x) in plans.items():
        record = json.loads((folder / 'complete.json').read_text())['arms'][arm]
        assert not record['refit_warnings']
        assert all(not f['warnings'] for c in record['sweep'] for f in c['folds'].values())
        path = folder / f'{arm}.joblib'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record['sha256']
        frozen[name] = {'path': str(path), 'sha256': digest, 'C': record['selected']['C']}
        p = np.full((len(rows), 2), np.nan)
        folds[name] = {}
        for held in DOMAINS:
            ti, ids = fold_indices(rows, held)
            model, warnings = fit_model(x, y, rows, ti, record['selected']['C'], max_iter=10000)
            if warnings:
                raise RuntimeError((name, held, warnings))
            assert model.classes_.tolist() == [0, 1]
            p[ids] = model.predict_proba(x[ids])
            folds[name][held] = score([rows[i] for i in ids], y[ids], p[ids].argmax(1))
            joblib.dump(model, ROOT / f'{name}_{held}.joblib')
            print(name, held, folds[name][held]['macro_dataset_accuracy'], flush=True)
        assert np.isfinite(p[vi]).all()
        probabilities[name] = p[vi]
        np.save(ROOT / f'{name}_held_probabilities.npy', p[vi])
    runs = []
    for alpha in [0., .25, .5, .75, 1.]:
        pred = mix(probabilities['covariance'], probabilities['reve'], alpha).argmax(1)
        runs.append({'alpha_reve': alpha, 'metrics': score(valid_rows, y[vi], pred)})
    selected = max(runs, key=lambda r: r['metrics']['macro_dataset_accuracy'])
    (ROOT / 'validation_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in valid_rows))
    (ROOT / 'source_complete.json').write_text(json.dumps({
        'selected': selected, 'sweep': runs, 'folds': folds, 'frozen': frozen,
        'covariance_provenance': cov_provenance, 'reve_provenance': reve_provenance,
        'target_evaluation': 'not_run',
        'note': 'Source-only development selection, not nested unbiased CV. Existing source-selected C fixed; smaller REVE weight wins exact ties.'
    }, indent=2))
    print('selected', selected, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
