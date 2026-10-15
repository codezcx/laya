"""Choose regularization by held-source validation; refit without target access."""
import hashlib
import json
import warnings
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .mi_covariance_v23 import ROOT, SOURCE, ARMS
from .source_probe import score

DOMAINS = ['Cho2017', 'Dreyer2023', 'Lee2019_MI']


def fold_indices(rows, held):
    train = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['dataset'] != held]
    validation = [i for i, r in enumerate(rows) if r['split'] == 'validation' and r['dataset'] == held]
    assert train and validation
    assert held not in {rows[i]['dataset'] for i in train}
    assert not {(rows[i]['dataset'], rows[i]['subject']) for i in train} & {
        (rows[i]['dataset'], rows[i]['subject']) for i in validation}
    return train, validation


def fit_model(x, y, rows, indices, c):
    counts = Counter((rows[i]['dataset'], int(y[i])) for i in indices)
    weights = np.array([1 / counts[(rows[i]['dataset'], int(y[i]))] for i in indices])
    weights /= weights.mean()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=42))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x[indices], y[indices], logisticregression__sample_weight=weights)
    return model, [str(w.message) for w in caught]


def load_cache(root=ROOT, source=SOURCE, arms=ARMS, dimension=30):
    manifest = source / 'manifest.jsonl'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original = [json.loads(line) for line in manifest.read_text().splitlines()]
    keep = np.load(source / 'keep_indices.npy')
    arrays = {arm: np.empty((len(original), dimension), np.float32) for arm in arms}
    seen, hashes = [], {}
    for shard in range(4):
        path = root / f'{shard}.npz'
        done = json.loads((root / f'{shard}.done.json').read_text())
        hashes[str(shard)] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert done['manifest_sha256'] == digest and done['feature_sha256'] == hashes[str(shard)]
        with np.load(path) as data:
            indices = data['indices']
            assert len(indices) == done['n']
            seen.extend(indices.tolist())
            for arm in arms:
                assert data[arm].shape == (len(indices), dimension) and np.isfinite(data[arm]).all()
                arrays[arm][indices] = data[arm]
    assert len(seen) == len(set(seen)) == 37432 and sorted(seen) == sorted(keep.tolist())
    rows = [original[int(i)] for i in keep]
    assert {r['dataset'] for r in rows} == set(DOMAINS)
    assert {r['split'] for r in rows} == {'train', 'validation'}
    train_subjects = {(r['dataset'], r['subject']) for r in rows if r['split'] == 'train'}
    val_subjects = {(r['dataset'], r['subject']) for r in rows if r['split'] == 'validation'}
    assert not train_subjects & val_subjects
    return rows, {k: v[keep] for k, v in arrays.items()}, {'manifest_sha256': digest, 'shards_sha256': hashes}


def main(root=ROOT, source=SOURCE, arms=ARMS, dimension=30):
    rows, features, provenance = load_cache(root, source, arms, dimension)
    out = root / 'fits'
    out.mkdir(exist_ok=False)
    (out / 'cache_audit.json').write_text(json.dumps(provenance, indent=2))
    y = np.array([r['label'] for r in rows])
    assert set(y) == {0, 1}
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    val = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    final = {}
    for arm in arms:
        x = features[arm]
        sweep = []
        for c in [.001, .01, .1, 1., 10.]:
            folds = {}
            for held in DOMAINS:
                ti, vi = fold_indices(rows, held)
                model, caught = fit_model(x, y, rows, ti, c)
                folds[held] = {'metrics': score([rows[i] for i in vi], y[vi], model.predict(x[vi])),
                               'warnings': caught, 'train_n': len(ti), 'validation_n': len(vi)}
            value = float(np.mean([v['metrics']['macro_dataset_accuracy'] for v in folds.values()]))
            sweep.append({'C': c, 'held_domain_mean': value, 'folds': folds})
            print(arm, c, value, flush=True)
        selected = max(sweep, key=lambda r: r['held_domain_mean'])
        model, caught = fit_model(x, y, rows, train, selected['C'])
        joblib.dump(model, out / f'{arm}.joblib')
        final[arm] = {'selected': selected, 'sweep': sweep, 'refit_warnings': caught,
                      'within_source_validation': score([rows[i] for i in val], y[val], model.predict(x[val])),
                      'sha256': hashlib.sha256((out / f'{arm}.joblib').read_bytes()).hexdigest()}
    (out / 'complete.json').write_text(json.dumps({
        'arms': final, 'provenance': provenance, 'train_n': len(train), 'validation_n': len(val),
        'target_evaluation': 'not_run', 'selection': 'Mean of three held-source validation folds; smaller C wins exact ties.'
    }, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
