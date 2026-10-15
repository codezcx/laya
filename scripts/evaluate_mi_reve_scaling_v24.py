"""Paired 2b evaluation, gated by converged source selection for both arms."""
import hashlib
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from eeg_tasks.data import preprocess
from eeg_tasks.mi_reve_scaling_v24 import ROOT, SOURCE, ARMS, scale_window
from eeg_tasks.source_probe import score


def main():
    fits = ROOT / 'fits_converged'
    complete = json.loads((fits / 'complete.json').read_text())
    frozen = {}
    for arm in ARMS:
        record = complete['arms'][arm]
        assert not record['refit_warnings']
        assert all(not f['warnings'] for c in record['sweep'] for f in c['folds'].values())
        frozen[arm] = hashlib.sha256((fits / f'{arm}.joblib').read_bytes()).hexdigest()
        assert frozen[arm] == record['sha256']
    out = ROOT / 'external_2b'
    out.mkdir(exist_ok=False)
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    manifest = Path('artifacts/reve_mi_multi/manifests_v2/external_2b.jsonl')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == len({r['path'] for r in rows}) == 2840
    source = [json.loads(line) for line in (SOURCE / 'manifest.jsonl').read_text().splitlines()]
    assert not {r['path'] for r in rows} & {r['path'] for r in source}
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4)
    base = json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    model = AutoModel.from_pretrained(base['reve_checkpoint'], local_files_only=True,
                                     trust_remote_code=True).cuda().eval().requires_grad_(False)
    bank = AutoModel.from_pretrained(base['positions_checkpoint'], local_files_only=True, trust_remote_code=True)
    pos = bank(['C3', 'Cz', 'C4']).detach().cuda()
    arrays = {arm: [] for arm in ARMS}
    for begin in range(0, len(rows), 64):
        signals = {arm: [] for arm in ARMS}
        for row in rows[begin:begin + 64]:
            with open(row['path'], 'rb') as f:
                item = pickle.load(f)
            assert item['freq'] == 200 and base['labels'].index(item['label']) == row['label']
            raw = np.stack([item['eeg'][c] for c in ['C3', 'CZ', 'C4']])
            window = preprocess(raw, 200, {'bandpass': [1, 30]})[:, :600]
            for arm in ARMS:
                signals[arm].append(scale_window(window, arm))
        for arm in ARMS:
            x = torch.from_numpy(np.stack(signals[arm])).cuda()
            stride = model.config.patch_size - model.config.patch_overlap
            tail = (-(600 - model.config.patch_size)) % stride
            x = torch.nn.functional.pad(x, (0, tail), mode='replicate')
            with torch.no_grad():
                z = model(x, pos[None].expand(len(x), -1, -1)).mean(2)
            assert torch.isfinite(z).all()
            arrays[arm].append(z.cpu().numpy().reshape(len(x), 1536).astype(np.float16))
        if begin % 512 == 0:
            print('extracted', begin, flush=True)
    y = np.array([r['label'] for r in rows])
    predictions, results = {}, {}
    for arm in ARMS:
        x = np.concatenate(arrays[arm]).astype(np.float32)
        pred = joblib.load(fits / f'{arm}.joblib').predict(x)
        predictions[arm] = pred
        results[arm] = score(rows, y, pred)
        results[arm]['trial_accuracy'] = float(np.mean(pred == y))
        results[arm]['shuffled'] = score(rows, y, pred[np.random.default_rng(2026).permutation(len(pred))])
        np.save(out / f'{arm}_predictions.npy', pred)
        np.save(out / f'{arm}_features.npy', x.astype(np.float16))
    subjects = sorted({r['subject'] for r in rows})
    delta = []
    for subject in subjects:
        ids = [i for i, r in enumerate(rows) if r['subject'] == subject]
        delta.append(float(np.mean(predictions['common_scale'][ids] == y[ids]) -
                           np.mean(predictions['per_channel'][ids] == y[ids])))
    boot = np.random.default_rng(2026).choice(delta, (10000, len(delta)), replace=True).mean(1)
    (out / 'results.json').write_text(json.dumps({
        'results': results, 'common_minus_per_channel': float(np.mean(delta)),
        'paired_subject_95ci': np.quantile(boot, [.025, .975]).tolist(), 'frozen': frozen,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'note': 'Exploratory repeated-target evaluation; REVE pretraining includes 2b. No target fitting or selection.'
    }, indent=2))
    print({arm: results[arm]['trial_accuracy'] for arm in ARMS}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
