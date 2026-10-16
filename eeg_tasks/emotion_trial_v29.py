"""Match evaluation context while comparing window and trial-level training loss."""
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .emotion_rating_v28 import split_indices
from .emotion_transfer_probe import metrics

ROOT = Path('artifacts/emotion_trial_v29')


def trial_groups(rows):
    groups = {}
    for i, r in enumerate(rows):
        key = (r['dataset'], r['subject'], r['path'], r['trial'])
        groups.setdefault(key, []).append(i)
    result = list(groups.values())
    for ids in result:
        if len({rows[i]['class_id'] for i in ids}) != 1 or len({rows[i]['split'] for i in ids}) != 1:
            raise ValueError('Conflicting trial labels or splits')
        if len({rows[i]['start_sample'] for i in ids}) != len(ids):
            raise ValueError('Repeated window within trial')
    return result


def main():
    ROOT.mkdir(exist_ok=False)
    previous = Path('artifacts/emotion_rating_v28/plan.json')
    plan = json.loads(previous.read_text())
    for key, digest in [('manifest', 'manifest_sha256'), ('feature_path', 'feature_sha256')]:
        assert hashlib.sha256(Path(plan[key]).read_bytes()).hexdigest() == plan[digest]
    rows = [json.loads(s) for s in Path(plan['manifest']).read_text().splitlines()]
    x = np.load(plan['feature_path'])
    ti, validation = split_indices(rows, set(plan['held_video_ids']))
    assert ti == plan['train_indices'] and validation == plan['validation_indices']
    groups = trial_groups(rows)
    trialrows = [dict(rows[ids[0]], windows=len(ids)) for ids in groups]
    trainset = set(ti)
    train = [j for j, ids in enumerate(groups) if set(ids) <= trainset]
    assert sum(len(groups[j]) for j in train) == len(ti)
    vals = {}
    for name, indices in validation.items():
        validset = set(indices)
        vals[name] = [j for j, ids in enumerate(groups) if set(ids) <= validset]
        assert sum(len(groups[j]) for j in vals[name]) == len(indices)
    means = np.stack([x[ids].mean(0) for ids in groups])
    # Both arms share the very same training-window scaler.
    scaler = StandardScaler().fit(x[ti])
    z, mz = scaler.transform(x), scaler.transform(means)
    labels = np.array([r['class_id'] for r in rows])
    triallabels = np.array([r['class_id'] for r in trialrows])
    counts = Counter((trialrows[j]['dataset'], int(triallabels[j])) for j in train)
    tw = np.array([1 / counts[(trialrows[j]['dataset'], int(triallabels[j]))] for j in train])
    tw *= len(train) / tw.sum()
    windowweight = np.zeros(len(rows))
    for j, weight in zip(train, tw):
        windowweight[groups[j]] = weight / len(groups[j])
    assert np.isclose(windowweight[ti].sum(), tw.sum())
    joblib.dump(scaler, ROOT / 'scaler.joblib')
    results = {}
    for arm in ['window_loss', 'trial_loss']:
        a, b, w = (z[ti], labels[ti], windowweight[ti]) if arm == 'window_loss' else (mz[train], triallabels[train], tw)
        sweep, best = [], -1.
        for c in [.001, .01, .1, 1., 10.]:
            model = LogisticRegression(C=c, max_iter=10000, random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(a, b, sample_weight=w)
            if caught:
                raise RuntimeError([str(q.message) for q in caught])
            scores = {name: metrics([trialrows[j] for j in ids], model.predict(mz[ids])) for name, ids in vals.items()}
            value = float(np.mean([scores[k]['subject_balanced_accuracy'] for k in ['SEED', 'FACED_unseen']]))
            record = {'C': c, 'selection_score': value, 'scores': scores};sweep.append(record)
            if value > best:
                best, selected = value, record;joblib.dump(model, ROOT / f'{arm}.joblib')
            print(arm, c, value, flush=True)
        results[arm] = {'selected': selected, 'sweep': sweep,
                        'sha256': hashlib.sha256((ROOT / f'{arm}.joblib').read_bytes()).hexdigest()}
    (ROOT / 'complete.json').write_text(json.dumps({'arms': results, 'train_trials': len(train),
        'validation_trials': {k: len(v) for k, v in vals.items()}, 'plan_sha256': hashlib.sha256(previous.read_bytes()).hexdigest(),
        'scaler_sha256': hashlib.sha256((ROOT / 'scaler.joblib').read_bytes()).hexdigest(),
        'target_evaluation': 'not_run', 'unit': 'Trial; both arms receive all retained trial windows at inference.'}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):main()
