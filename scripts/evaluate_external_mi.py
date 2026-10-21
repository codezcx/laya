"""Fixed-checkpoint external MI evaluation; no target training or selection."""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel
from eeg_align.jev import load_jev, JEVBridge
from eeg_align.mi_training_data import canonical_channel, digest, metrics, position_names, readrows, write_json

EXPECTED = {'Beetl2021_A': (100, 2), 'Beetl2021_B': (180, 3), 'Shin2017A': (1740, 29)}
ROOT = Path('/inspire/dataset/moabb-processed/v1/EEG-processed')


def prepare(out):
    """Audit all retained channels and persist exact inference inputs/provenance."""
    names, lookup = position_names()
    sources = readrows('datasets/mi_tokens/train.jsonl') + readrows('datasets/mi_tokens/validation.jsonl')
    assert not set(EXPECTED) & {r['dataset'] for r in sources}
    source_hashes = {r['signal_sha256'] for r in sources}
    seen = set()
    arrays, rows = [], []
    for ds, (count, subjects) in EXPECTED.items():
        selected = []
        for path in sorted((ROOT / ds).rglob('*.pkl')):
            label = path.stem.rsplit('_label-', 1)[-1]
            if label not in ('left_hand', 'right_hand'):
                continue
            with path.open('rb') as f:
                r = pickle.load(f)
            assert r['dataset'] == ds and r['label'] == label
            assert r['schema_version'] == 'eeg-epoch-v3' and r['paradigm'] == 'imagery'
            assert r['unit'] == 'uV' and float(r['freq']) == 200
            channels = list(r['eeg'])
            mapped = [lookup[canonical_channel(c)] for c in channels]
            assert len(set(mapped)) == len(mapped)
            x = np.stack([r['eeg'][c] for c in channels]).astype(np.float32)
            assert x.shape == (len(channels), r['n_samples'])
            assert np.isfinite(x).all() and (x.std(-1) >= 1e-8).all()
            full_hash = hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()
            assert full_hash not in seen, 'Duplicate external signal'
            seen.add(full_hash)
            normalized = {canonical_channel(c): i for i, c in enumerate(channels)}
            if all(c in normalized for c in ('C3', 'CZ', 'C4')):
                h = hashlib.sha256(x[[normalized[c] for c in ('C3', 'CZ', 'C4')]].astype('<f4').tobytes()).hexdigest()
                assert h not in source_hashes, 'Signal overlap with source train/validation'
            n = x.shape[-1]
            assert n >= 200
            x -= x.mean(-1, keepdims=True)
            clipped = float((abs(x) > 1000).mean())
            x = np.clip(x, -1000, 1000) / 100
            target = 200 + math.ceil((n - 200) / 180) * 180
            x = np.pad(x, ((0, 0), (0, target - n)), mode='edge')
            row = dict(id=hashlib.sha256(str(path).encode()).hexdigest()[:24], path=str(path),
                       file_sha256=digest(path), dataset=ds, subject=str(r['subject']),
                       session=str(r['session']), label=int(label == 'right_hand'), channels=channels,
                       position_names=mapped, original_samples=n, padded_samples=target,
                       token_length=len(channels)*(1+(target-200)//180), clip_fraction=clipped,
                       signal_sha256=full_hash, raw_name=r.get('raw_name'))
            rows.append(row); arrays.append(x); selected.append(row)
        assert len(selected) == count and len({r['subject'] for r in selected}) == subjects
        print('AUDIT', ds, count, dict(Counter(r['label'] for r in selected)), flush=True)
    out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); torch.set_num_threads(3)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    if rank == 0: args.out.mkdir(parents=True, exist_ok=False)
    dist.barrier(); began = time.monotonic()
    checkpoint = Path('artifacts/mi_jev_tuab_lr_continued/best.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['step'] == 2307
    for f in ['eeg_align/jev.py', 'eeg_align/mi_training_data.py', 'configs/encoder.json']:
        assert digest(f) == saved['config']['code_sha256'][f], f
    # Each rank independently audits identical input; manifests record each audit.
    rows, arrays = prepare(args.out / f'manifest_rank{rank}.jsonl')
    ec = json.loads(Path('configs/encoder.json').read_text())
    bank = AutoModel.from_pretrained(ec['positions_checkpoint'], trust_remote_code=True, local_files_only=True).eval()
    reve, info = AutoModel.from_pretrained(ec['reve_checkpoint'], trust_remote_code=True, local_files_only=True, output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
    reve = reve.cuda().eval().requires_grad_(False)
    ids = list(range(rank, len(rows), world)); features = []
    with torch.no_grad():
        for i in ids:
            x = torch.from_numpy(arrays[i]).cuda()[None]
            positions = bank(rows[i]['position_names']).detach().cuda()[None]
            z = reve(x, positions).flatten(1, 2)[0]
            assert z.shape == (rows[i]['token_length'], 512) and torch.isfinite(z).all()
            features.append(z.half().cpu())
    del reve, bank, arrays; torch.cuda.empty_cache()
    text, w, b, tok = load_jev()
    model = JEVBridge(text, w, b, tok, train_lora=True, train_head=True)
    assert set(saved['trainable']) == {n for n, p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(saved['trainable'], strict=False); del saved
    model.eval().requires_grad_(False)
    local = []
    with torch.no_grad():
        for j, (i, feature) in enumerate(zip(ids, features)):
            z = feature.cuda()[None]; valid = torch.ones(z.shape[:2], device='cuda', dtype=torch.bool)
            probability = model(z, valid)['logits'].softmax(-1)[0, 1].item()
            local.append(dict(rows[i], probability_right=probability))
            if j % 50 == 0: print('EVAL', rank, j, len(ids), flush=True)
    parts = [None]*world; dist.all_gather_object(parts, local)
    if rank == 0:
        pred = sorted(sum(parts, []), key=lambda r: r['id'])
        assert len(pred) == 2020 and len({r['id'] for r in pred}) == 2020
        manifests = [digest(args.out/f'manifest_rank{r}.jsonl') for r in range(world)]
        assert len(set(manifests)) == 1
        write_json(args.out/'config.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), step=2307,
            script_sha256=digest(__file__), manifest_sha256=manifests[0], encoder=ec, threshold=.5,
            preprocessing='Source-matched 200Hz uV; channel centering; clip +/-1000; /100; edge-pad partial final patch; all EEG channels',
            overlap_check='Dataset exclusion; C3/Cz/C4 exact signal hashes where available (Beetl); Shin lacks this montage, no source exact-signal comparison claimed',
            protocol='Fixed source-selected weights, no target tuning; original A=left B=right prompt'))
        write_json(args.out/'predictions.json', pred)
        result = metrics(pred); write_json(args.out/'metrics.json', result)
        write_json(args.out/'complete.json', dict(n=len(pred), elapsed_seconds=time.monotonic()-began))
        print('RESULT', {k:v['accuracy'] for k,v in result['datasets'].items()}, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
