"""Real TRAIN-only encoder/JEV forward-backward check, never reused as SFT weights."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG
from eeg_align.benchmark_data import catalog
from eeg_align.multitask_data import SignalAdapter, write_json
from eeg_align.online_leaf import OnlineLEAF
from eeg_align.jev import JEVBridge, load_jev
from eeg_align.task_options import make_variant
from eeg_align.checkpoint import seed_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    seed_rng(42)
    tasks = catalog()
    adapter = SignalAdapter()
    text, weights, bias, tokenizer = load_jev()
    model = JEVBridge(text, weights, bias, tokenizer, True, True, head_choices=16, input_dim=256)
    model.add_module('eeg_encoder', OnlineLEAF().cuda())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    report = {}
    for tid, task in enumerate(tasks[:9]):
        candidates = sorted(Path('datasets').glob(f'multitask_20261029_audit*/{task["task_id"]}/index.csv'))
        if not candidates:
            raise FileNotFoundError(task['task_id'])
        source = candidates[-1]
        rows = pd.read_csv(source, dtype={'subject': str}).query('split == "train"').iloc[:2].to_dict('records')
        for row in rows:
            row.update(task=tid, array=str((source.parent / row['array']).resolve()))
        question = make_variant(task, rows[0]['label'], 'train-preflight', 0, 42, 0)
        suffix = torch.tensor(tokenizer.encode(question['suffix'], add_special_tokens=False), device='cuda')
        target = torch.tensor([r['label'] for r in rows], device='cuda')
        start = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tokens, valid = model.eeg_encoder(adapter.batch(rows), len(rows), unfrozen=True)
        value = model(tokens, valid, labels=target, suffix_ids=[suffix] * len(rows), num_choices=len(task['classes']))
        if not torch.isfinite(value['loss']):
            raise FloatingPointError('Nonfinite preflight loss')
        value['loss'].backward()
        active = [p for p in model.parameters() if p.grad is not None]
        grad = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
        encoder_grad = sum(float(p.grad.float().square().sum()) for p in model.eeg_encoder.parameters() if p.grad is not None) ** .5
        if encoder_grad <= 0:
            raise ValueError('No gradient reaching LEAF')
        optimizer.step()
        report[task['task_id']] = dict(split='train', n=len(rows), tokens=list(tokens.shape), loss=float(value['loss'].detach()), gradient_norm=float(grad), encoder_gradient_norm=encoder_grad, seconds=time.monotonic() - start, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        print('FORWARD_BACKWARD_OK', task['task_id'], report[task['task_id']], flush=True)
        write_json(args.out / 'progress.json', report)
    write_json(args.out / 'complete.json', dict(datasets=report, test_evaluations=0, weights_discarded=True))


if __name__ == '__main__':
    main()
