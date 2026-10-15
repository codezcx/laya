"""Repair only warned source folds, retaining the original v24 selection record."""
import hashlib
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from eeg_tasks.mi_covariance_fit_v23 import load_cache, fold_indices, fit_model
from eeg_tasks.mi_reve_scaling_v24 import ROOT, SOURCE, ARMS
from eeg_tasks.source_probe import score


def main():
    original = ROOT / 'fits/complete.json'
    result = json.loads(original.read_text())
    rows, features, provenance = load_cache(ROOT, SOURCE, ARMS, 1536)
    assert provenance == result['provenance']
    out = ROOT / 'fits_converged'
    out.mkdir(exist_ok=False)
    result['original_complete_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
    result['repair'] = 'Only warned folds recomputed at max_iter=10000, unchanged C grid and selection. Both final models refit; originals preserved.'
    y = np.array([r['label'] for r in rows])
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    val = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    for arm in ARMS:
        record = result['arms'][arm]
        x = features[arm]
        for candidate in record['sweep']:
            for held, fold in candidate['folds'].items():
                if not fold['warnings']:
                    continue
                ti, vi = fold_indices(rows, held)
                model, caught = fit_model(x, y, rows, ti, candidate['C'], max_iter=10000)
                if caught:
                    raise RuntimeError((arm, candidate['C'], held, caught))
                fold['metrics'] = score([rows[i] for i in vi], y[vi], model.predict(x[vi]))
                fold['original_warnings'] = fold['warnings']
                fold['warnings'] = []
                fold['max_iter'] = 10000
                print(arm, candidate['C'], held, fold['metrics']['macro_dataset_accuracy'], flush=True)
            candidate['held_domain_mean'] = float(np.mean([f['metrics']['macro_dataset_accuracy'] for f in candidate['folds'].values()]))
        record['selected'] = max(record['sweep'], key=lambda c: c['held_domain_mean'])
        model, caught = fit_model(x, y, rows, train, record['selected']['C'], max_iter=10000)
        if caught:
            raise RuntimeError(caught)
        joblib.dump(model, out / f'{arm}.joblib')
        record['sha256'] = hashlib.sha256((out / f'{arm}.joblib').read_bytes()).hexdigest()
        record['refit_warnings'] = []
        record['within_source_validation'] = score([rows[i] for i in val], y[val], model.predict(x[val]))
    (out / 'complete.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
