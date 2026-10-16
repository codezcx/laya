"""Evaluate both frozen raw-power representations on identical SEED-VII windows."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from .data import preprocess
from .emotion_power_v26 import ROOT, power_features
from .emotion_transfer_probe import picks, readrows, metrics, CHANNELS

CACHE = Path('artifacts/emotion_alignment_v21/external_seed_vii')
OUT = ROOT / 'external_seed_vii'
ARMS = ['per_channel', 'joint_power']


def freeze():
    frozen = json.loads((ROOT / 'source_freeze.json').read_text())['models']
    complete = json.loads((ROOT / 'fits/complete.json').read_text())
    assert set(frozen) == set(ARMS)
    for a in ARMS:
        assert hashlib.sha256((ROOT / 'fits' / f'{a}.joblib').read_bytes()).hexdigest() == frozen[a] == complete['arms'][a]['sha256']
    return frozen


def extract(shard):
    frozen = freeze()
    OUT.mkdir(exist_ok=True)
    dest = OUT / f'{shard}.npz'
    if dest.exists():
        raise FileExistsError(dest)
    rows = readrows(CACHE / f'rows_{shard}.jsonl')
    assert len(rows) == 1600 and {r['dataset'] for r in rows} == {'SEED-VII-HNS'}
    names_path = Path(rows[0]['path']).parent.parent / 'channel_62_pos.locs'
    names = [s.split()[-1] for s in names_path.read_text().splitlines() if s.strip()]
    cp = picks(names)
    assert json.loads((CACHE / f'{shard}.done.json').read_text())['channels'] == CHANNELS
    features = {a: np.empty((len(rows), 150), np.float32) for a in ARMS}
    for path in sorted({r['path'] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r['path'] == path]
        trials = sorted({rows[i]['trial'] for i in ix})
        data = loadmat(path, variable_names=trials)
        for trial in trials:
            x = preprocess(data[trial][cp], 200, {'bandpass': [1, 30]})
            for i in [i for i in ix if rows[i]['trial'] == trial]:
                start = rows[i]['start_sample']
                w = x[:, start:start + 600]
                for a in ARMS:
                    features[a][i] = power_features(w, a).reshape(150)
        print(shard, Path(path).name, flush=True)
    reference = np.load(CACHE / f'spectral_{shard}.npy')
    error = float(np.abs(features['per_channel'] - reference).max())
    assert error < 1e-5, error
    np.savez(dest, **features)
    (OUT / f'{shard}.done.json').write_text(json.dumps({
        'frozen': frozen, 'n': len(rows), 'control_replay_max_abs': error,
        'rows_sha256': hashlib.sha256((CACHE / f'rows_{shard}.jsonl').read_bytes()).hexdigest(),
        'feature_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
    }, indent=2))


def evaluate():
    frozen = freeze()
    if (OUT / 'results.json').exists():
        raise FileExistsError(OUT / 'results.json')
    rows, arrays = [], {a: [] for a in ARMS}
    for shard in range(4):
        done = json.loads((OUT / f'{shard}.done.json').read_text())
        assert done['frozen'] == frozen
        assert done['rows_sha256'] == hashlib.sha256((CACHE / f'rows_{shard}.jsonl').read_bytes()).hexdigest()
        assert done['feature_sha256'] == hashlib.sha256((OUT / f'{shard}.npz').read_bytes()).hexdigest()
        rows.extend(readrows(CACHE / f'rows_{shard}.jsonl'))
        with np.load(OUT / f'{shard}.npz') as data:
            for a in ARMS:
                assert data[a].shape == (done['n'], 150) and np.isfinite(data[a]).all()
                arrays[a].append(data[a])
    assert len(rows) == 6400 and len({(r['subject'], r['trial'], r['start_sample']) for r in rows}) == 6400
    assert len({r['subject'] for r in rows}) == 20
    results = {}
    for a in ARMS:
        pred = joblib.load(ROOT / 'fits' / f'{a}.joblib').predict(np.concatenate(arrays[a]))
        results[a] = metrics(rows, pred)
        results[a]['shuffled'] = metrics(rows, pred[np.random.default_rng(2026).permutation(len(pred))])
        np.save(OUT / f'{a}_predictions.npy', pred)
    x, y = [results[a]['per_subject'] for a in ARMS]
    delta = np.array([y[s]['balanced_accuracy'] - x[s]['balanced_accuracy'] for s in sorted(x)])
    boot = np.random.default_rng(2026).choice(delta, (10000, len(delta)), replace=True).mean(1)
    (OUT / 'results.json').write_text(json.dumps({'results': results, 'frozen': frozen,
        'joint_minus_per_channel': float(delta.mean()), 'paired_subject_95ci': np.quantile(boot, [.025, .975]).tolist(),
        'note': 'Repeated-target exploratory HNS evaluation; no target fitting or model choice.'}, indent=2))
    print({a: results[a]['subject_balanced_accuracy'] for a in ARMS}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, choices=range(4))
    p.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.evaluate:
            evaluate()
        elif args.shard is not None:
            extract(args.shard)
        else:
            p.error('Choose extraction shard or evaluation')
