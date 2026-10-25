"""Known-answer controls for JEV option binding; never EEG accuracy estimates.

Compare native and adapted JEV on task/class text with cyclic option orders, then
check adapted JEV with actual source EEG, with/without an explicit category hint.
Only deterministic training examples are used; no optimizer or training mutation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from eeg_align.benchmark_data import catalog
from eeg_align.checkpoint import capture_model, restore_model, seed_rng, tensor_fingerprint
from eeg_align.jev import JEVBridge, assemble_embeddings, choice_logits, load_jev
from eeg_align.leaf_runtime import CONFIG, digest
from eeg_align.online_leaf import OnlineLEAF
from eeg_align.signal_data import ROOT as SIGNAL_ROOT, SignalStore, load_rows


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def make_cases(tasks, rows):
    selected, cases = [], []
    for tid, task in enumerate(tasks):
        count = len(task['classes'])
        for label, cls in enumerate(task['classes']):
            row = next(r for r in rows if r['task'] == tid and r['label'] == label)
            assert row['split'] == 'train'
            row_index = len(selected)
            selected.append(row)
            hint = f"Known category: {cls['option']}. Use this supplied category to choose the matching option.\n"
            for rotation in range(count):
                order = list(range(rotation, count)) + list(range(rotation))
                options = '\n'.join(f"{chr(65+i)}) {task['classes'][c]['option']}" for i, c in enumerate(order))
                question = f"[question] {task['question']}\n[options]\n{options}\n[decision]:"
                cases.append(dict(dataset=task['task_id'], sample=row['id'], row_index=row_index,
                                  true_class=label, rotation=rotation, order=order,
                                  target_slot=order.index(label), num_choices=count,
                                  native_text='[kind] choice\n[state] ' + hint + question,
                                  suffix='\nEEG recording ends.\n' + question,
                                  oracle_suffix='\nEEG recording ends.\n' + hint + question))
        cc = [c for c in cases if c['dataset'] == task['task_id']]
        assert len(cc) == count**2
        for label in range(count):
            assert sorted(c['target_slot'] for c in cc if c['true_class'] == label) == list(range(count))
        for rotation in range(count):
            assert len({tuple(c['order']) for c in cc if c['rotation'] == rotation}) == 1
    return selected, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    assert args.batch_size > 0
    torch.set_num_threads(3)
    torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0] > 65 * 2**30, 'Insufficient spare GPU memory for isolated JEV inference'
    seed_rng(42)
    before = args.checkpoint.stat()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert checkpoint['step'] == args.expected_step and checkpoint['format_version'] == 2
    assert checkpoint['config']['bn_policy'] == 'frozen'
    for key, path in [
        ('task_sha256', Path('configs/benchmark_tasks.json')),
        ('instructions_sha256', Path('artifacts/benchmark/instructions.npy')),
        ('data_manifest_sha256', SIGNAL_ROOT / f"fold{CONFIG['fold']}" / 'complete.json'),
    ]:
        assert checkpoint['config'][key] == digest(path)
    for path, expected in checkpoint['config']['code_sha256'].items():
        assert digest(path) == expected, path
    tasks = catalog()
    rows, cases = make_cases(tasks, load_rows('train', CONFIG['fold']))
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'cases.json', cases)
    write_json(args.out / 'plan.json', dict(
        checkpoint=str(args.checkpoint.resolve()), step=args.expected_step, source_rows=rows,
        source_split='train', cases=len(cases), examples=len(rows), batch_size=args.batch_size,
        cases_sha256=digest(args.out / 'cases.json'), script_sha256=digest(__file__),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        native_template='bare-v1, checked against local JEV README/server',
        conditions=['native/native_text', 'native/bridge_text', 'adapted/native_text',
                    'adapted/bridge_text', 'adapted/signal_only', 'adapted/signal_oracle'],
        interpretation='The correct category is intentionally supplied in oracle conditions. '
                       'These are option-following controls, never EEG generalization scores. '
                       'Signal-only uses only one source example per class and is not a training-set accuracy estimate. '
                       'Uncalibrated logits are used; a positive shared temperature would not change argmax.'))
    text, weight, bias, tokenizer = load_jev()
    model = JEVBridge(text, weight, bias, tokenizer, True, True, head_choices=16, input_dim=256)
    model.add_module('eeg_encoder', OnlineLEAF('frozen').cuda())
    model.eval()
    started = time.monotonic()
    results = {}
    predictions = []
    encode = lambda s: torch.tensor(tokenizer.encode(s, add_special_tokens=False), device='cuda')
    tokens, valid = None, None

    @torch.inference_mode()
    def run(stage, condition):
        records = []
        for start in range(0, len(cases), args.batch_size):
            chunk = cases[start:start + args.batch_size]
            counts = [c['num_choices'] for c in chunk]
            if condition.startswith('signal_'):
                indices = [c['row_index'] for c in chunk]
                key = 'oracle_suffix' if condition == 'signal_oracle' else 'suffix'
                logits = model(tokens[indices], valid[indices], suffix_ids=[encode(c[key]) for c in chunk],
                               num_choices=counts)['logits']
            else:
                embed = model.text.get_input_embeddings()
                if condition == 'native_text':
                    prefix = embed(model.prefix_ids[:0])
                    suffixes = [embed(encode(c['native_text'])) for c in chunk]
                else:
                    prefix = embed(model.prefix_ids)
                    suffixes = [embed(encode(c['oracle_suffix'])) for c in chunk]
                empty = torch.empty((len(chunk), 0, 5120), device='cuda', dtype=prefix.dtype)
                mask = torch.empty((len(chunk), 0), device='cuda', dtype=torch.bool)
                x, mask, positions, last = assemble_embeddings(prefix, empty, suffixes, mask)
                hidden = model.text(inputs_embeds=x, attention_mask=mask, position_ids=positions,
                                    use_cache=False, return_dict=True).last_hidden_state
                state = hidden[torch.arange(len(chunk), device='cuda'), last]
                logits = choice_logits(state.float(), model.head_weight, model.head_bias, counts)
            probabilities = logits.softmax(-1).cpu().numpy()
            for case, p in zip(chunk, probabilities):
                p = p[:case['num_choices']]
                assert np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-6)
                slot = int(p.argmax())
                records.append(dict(dataset=case['dataset'], sample=case['sample'], rotation=case['rotation'],
                                    target_slot=case['target_slot'], predicted_slot=slot,
                                    semantic_prediction=case['order'][slot], true_class=case['true_class'],
                                    correct=slot == case['target_slot'], probabilities=p.tolist()))
        summary = {}
        for task in tasks:
            ds = task['task_id']; rr = [r for r in records if r['dataset'] == ds]
            by_class = [[r for r in rr if r['true_class'] == label] for label in range(len(task['classes']))]
            summary[ds] = dict(n=len(rr), accuracy=float(np.mean([r['correct'] for r in rr])),
                               all_rotations_correct_classes=sum(all(r['correct'] for r in group) for group in by_class),
                               classes=len(by_class), predicted_slot_counts=np.bincount(
                                   [r['predicted_slot'] for r in rr], minlength=len(by_class)).tolist())
        results[f'{stage}/{condition}'] = summary
        predictions.extend(dict(stage=stage, condition=condition, **r) for r in records)
        write_json(args.out / 'metrics.json', results)
        print(json.dumps(dict(stage=stage, condition=condition, accuracies={ds:v['accuracy'] for ds,v in summary.items()})), flush=True)

    for condition in ['native_text', 'bridge_text']:
        run('native', condition)
    restore_model(model, checkpoint)
    state = capture_model(model)
    fingerprints = {k:tensor_fingerprint(state[k]) for k in ['trainable','buffers']}
    assert all(fingerprints[k] == tensor_fingerprint(checkpoint[k]) for k in fingerprints)
    torch.save(state, args.out / 'model_state.pt')
    after = args.checkpoint.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    del checkpoint, state
    model.eval()  # Keep requires_grad flags for exact before/after state fingerprints; inference_mode disables grads.
    store = SignalStore()
    with torch.inference_mode():
        chunks, masks = [], []
        for start in range(0, len(rows), args.batch_size):
            rr = rows[start:start + args.batch_size]
            t, m = model.eeg_encoder(store.batch(rr), len(rr), unfrozen=False)
            chunks.append(t); masks.append(m)
        tokens, valid = torch.cat(chunks), torch.cat(masks)
    for condition in ['native_text', 'bridge_text', 'signal_only', 'signal_oracle']:
        run('adapted', condition)
    after = capture_model(model)
    assert all(tensor_fingerprint(after[k]) == fingerprints[k] for k in fingerprints)
    (args.out / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    write_json(args.out / 'complete.json', dict(
        elapsed_seconds=time.monotonic() - started, model_state_unchanged=True,
        restored_state_sha256=fingerprints, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        cases_per_condition=len(cases), source_examples=len(rows)))


if __name__ == '__main__':
    main()
