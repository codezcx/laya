"""Matched subjective-rating auxiliary controls with held subjects and videos."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('artifacts/emotion_rating_v28')
ARMS = ['classification', 'rating', 'shuffled_rating', 'class_mean_rating']


def split_indices(rows, held):
    train = [i for i, r in enumerate(rows) if r['split'] == 'train' and
             (r['dataset'] == 'SEED' or int(r['trial']) not in held)]
    validation = {
        'SEED': [i for i, r in enumerate(rows) if r['split'] == 'validation' and r['dataset'] == 'SEED'],
        'FACED_unseen': [i for i, r in enumerate(rows) if r['split'] == 'validation' and r['dataset'] == 'FACED-NEMAR' and int(r['trial']) in held],
        'FACED_seen': [i for i, r in enumerate(rows) if r['split'] == 'validation' and r['dataset'] == 'FACED-NEMAR' and int(r['trial']) not in held],
    }
    assert train and all(validation.values())
    assert not {(rows[i]['dataset'], rows[i]['subject']) for i in train} & {
        (rows[i]['dataset'], rows[i]['subject']) for ids in validation.values() for i in ids}
    return train, validation


def prepare():
    ROOT.mkdir(exist_ok=False)
    manifest = Path('artifacts/faced_scale_v22/rows.jsonl')
    rating_path = Path('artifacts/emotion_supervision_v27/ratings.jsonl')
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    ratings = [json.loads(s) for s in rating_path.read_text().splitlines()]
    # Determine stimulus identities/categories from metadata, not rating values or outcomes.
    categories = {}
    for r in ratings:
        categories.setdefault(r['target_emotion'], set()).add(r['video'])
    assert len(categories) == 9
    rng = np.random.default_rng(2801)
    held = sorted(int(rng.choice(sorted(v))) for k, v in sorted(categories.items()))
    assert len(set(held)) == 9
    train, val = split_indices(rows, set(held))
    keys = sorted({(int(rows[i]['subject']), int(rows[i]['trial'])) for i in train if rows[i]['dataset'] == 'FACED-NEMAR'})
    lookup = {(r['subject'], r['video']): r for r in ratings}
    assert len(lookup) == len(ratings)
    values = np.array([lookup[k]['ratings']['Valence'] for k in keys], np.float64)
    assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 7))
    classes = np.array([{'negative': 0, 'neutral': 1, 'positive': 2}[lookup[k]['stimulus_label']] for k in keys])
    shuffled, means = values.copy(), values.copy()
    rng = np.random.default_rng(2802)
    for c in range(3):
        ix = np.flatnonzero(classes == c)
        shuffled[ix] = values[rng.permutation(ix)]
        means[ix] = values[ix].mean()
    mu, sd = float(values.mean()), float(values.std())
    assert sd > 0
    targets = {a: np.zeros(len(rows), np.float32) for a in ARMS}
    target_lookup = {a: dict(zip(keys, (v - mu) / sd)) for a, v in
                     [('rating', values), ('shuffled_rating', shuffled), ('class_mean_rating', means)]}
    for i in train:
        r = rows[i]
        if r['dataset'] == 'FACED-NEMAR':
            k = (int(r['subject']), int(r['trial']))
            for a in target_lookup:
                targets[a][i] = target_lookup[a][k]
    np.savez(ROOT / 'targets.npz', **targets)
    feature_path = Path('artifacts/emotion_power_v26/per_channel.npy')
    assert np.load(feature_path, mmap_mode='r').shape == (len(rows), 150)
    (ROOT / 'plan.json').write_text(json.dumps({
        'held_video_ids': held, 'train_indices': train, 'validation_indices': val,
        'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'rating_source_sha256': hashlib.sha256(rating_path.read_bytes()).hexdigest(),
        'feature_path': str(feature_path), 'feature_sha256': hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        'rating_mean': mu, 'rating_std': sd, 'unique_training_rating_trials': len(keys),
        'steps': 2000, 'batch_size': 192, 'seed': 42, 'aux_weight': .2,
        'note': 'One video per fine emotion held from all fitting; only source validation subjects scored. No target access.'
    }, indent=2))
    print('held videos', held, 'train', len(train), 'validation', {k: len(v) for k, v in val.items()}, flush=True)


def train(arm):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .emotion_transfer_probe import metrics
    torch.set_num_threads(4)
    plan = json.loads((ROOT / 'plan.json').read_text())
    for path_key, hash_key in [('manifest', 'manifest_sha256'), ('feature_path', 'feature_sha256')]:
        assert hashlib.sha256(Path(plan[path_key]).read_bytes()).hexdigest() == plan[hash_key]
    rows = [json.loads(s) for s in Path(plan['manifest']).read_text().splitlines()]
    ti, val = split_indices(rows, set(plan['held_video_ids']))
    assert ti == plan['train_indices'] and val == plan['validation_indices']
    x = np.load(plan['feature_path']).astype(np.float32)
    mu, sd = x[ti].mean(0), np.maximum(x[ti].std(0), 1e-6)
    x = torch.from_numpy((x - mu) / sd).cuda()
    y = torch.tensor([r['class_id'] for r in rows], device='cuda')
    target = torch.from_numpy(np.load(ROOT / 'targets.npz')[arm]).cuda()
    faced = torch.tensor([r['dataset'] == 'FACED-NEMAR' for r in rows], device='cuda')
    groups = [[i for i in ti if rows[i]['dataset'] == ds and rows[i]['class_id'] == c]
              for ds in ['SEED', 'FACED-NEMAR'] for c in range(3)]
    assert all(groups)
    torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    trunk = nn.Sequential(nn.Linear(150, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 64), nn.GELU()).cuda()
    classifier, regressor = nn.Linear(64, 3).cuda(), nn.Linear(64, 1).cuda()
    model = nn.ModuleDict({'trunk': trunk, 'classifier': classifier, 'regressor': regressor})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    rng = np.random.default_rng(plan['seed'])
    out = ROOT / arm;out.mkdir(exist_ok=False)
    np.savez(out / 'normalizer.npz', mean=mu, std=sd)
    best, histories = -1., []
    for step in range(plan['steps'] + 1):
        if step % 100 == 0:
            model.eval();scores = {}
            with torch.no_grad():
                for name, ids in val.items():
                    pred = torch.cat([classifier(trunk(x[b])).argmax(1).cpu() for b in np.array_split(ids, max(1, len(ids)//512))]).numpy()
                    scores[name] = metrics([rows[i] for i in ids], pred)
            value = float(np.mean([scores[k]['subject_balanced_accuracy'] for k in ['SEED', 'FACED_unseen']]))
            histories.append({'step': step, 'selection_score': value, 'scores': scores})
            if value > best:
                best = value;selected = histories[-1];torch.save(model.state_dict(), out / 'best.pt')
            print(arm, step, value, flush=True)
            (out / 'validation.json').write_text(json.dumps(histories, indent=2))
        if step == plan['steps']:
            break
        model.train();ids = np.concatenate([rng.choice(g, 32, replace=True) for g in groups])
        state = trunk(x[ids]);ce = F.cross_entropy(classifier(state), y[ids])
        mask = faced[ids]
        auxiliary = F.smooth_l1_loss(regressor(state[mask]).squeeze(1), target[ids][mask])
        loss = ce + (0 if arm == 'classification' else plan['aux_weight']) * auxiliary
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
        if step % 50 == 0:
            with (out / 'loss.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'ce': ce.item(), 'auxiliary': auxiliary.item(), 'total': loss.item()}) + '\n')
    (out / 'complete.json').write_text(json.dumps({'arm': arm, 'selected': selected,
        'sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest(),
        'plan_sha256': hashlib.sha256((ROOT / 'plan.json').read_bytes()).hexdigest(),
        'target_evaluation': 'not_run'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--prepare', action='store_true');p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    if a.prepare:prepare()
    elif a.arm:train(a.arm)
    else:p.error('Choose stage')
