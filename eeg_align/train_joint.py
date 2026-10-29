"""One shared LEAF/JEV model on locked multitask subjects; full final evaluation."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .runtime import CONFIG, digest
from .benchmark_data import catalog, classification_metrics
from .multitask_data import TrialStore, SignalAdapter, write_json
from .online_leaf import OnlineLEAF
from .jev import load_jev, JEVBridge
from .task_options import make_variant
from .checkpoint import capture_model, restore_model, seed_rng, capture_rng, restore_rng, tensor_fingerprint
from .distributed import average_gradients, atomic_save
from .lr_schedule import GroupLRSchedule


def require_test(step, total, smoke):
    if smoke or step != total:
        raise RuntimeError('Test access forbidden before fixed training budget finishes')


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    device = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(3)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=torch.device('cuda', device))
    def barrier():
        if world > 1:
            dist.barrier()
    def collect(value):
        if world == 1:
            return [value]
        result = [None] * world
        dist.all_gather_object(result, value)
        return result
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if rank == 0:
        if args.resume:
            if not (args.out / 'last.pt').exists() or (args.out / 'complete.json').exists():
                raise ValueError('Resume requires unfinished run with last.pt')
        else:
            args.out.mkdir(parents=True, exist_ok=False)
    barrier()
    store = TrialStore()
    adapter = SignalAdapter()
    tasks = catalog()
    total = 2 if args.smoke else CONFIG['steps']
    freeze = 1 if args.smoke else CONFIG['encoder_warmup_steps']
    batch_size = CONFIG['global_batch']
    if batch_size % world:
        raise ValueError('Global batch must divide world size')
    leaf_root = Path(CONFIG['leaf_root'])
    watch = sorted(list(Path('eeg_align').glob('*.py')) + [Path('configs/benchmark_train.json'), Path('configs/benchmark_tasks.json'), Path('scripts/prepare_multitask_signals.py')] +
        [leaf_root / p for p in ['LEAF.py', 'load_config.py', 'utils.py', 'configs/LEAF_mpnet.yaml', 'configs/env.yaml', 'leaf_datasets/electrode_unifier.py', 'configs/LEAF-ch65/LEAF-ch65.json', 'configs/LEAF-ch65/LEAF-ch65.bvef']])
    hashes = {str(p): digest(p) for p in watch}
    def guard():
        if hashes != {str(p): digest(p) for p in watch}:
            raise RuntimeError('Training code/recipe changed during live run')
    identity = dict(config=CONFIG, smoke=args.smoke, world_size=world, code_sha256=hashes,
                    checkpoint_sha256=digest(CONFIG['checkpoint']),
                    instructions_sha256=digest(Path(CONFIG['text_root']) / 'instructions.npy'),
                    cache_indexes={tasks[t]['task_id']: m['index_sha256'] for t, m in store.meta.items()})
    import transformers, mne
    jev_root = Path('laya/JEV-27B-VL')
    identity['runtime'] = dict(torch=torch.__version__, transformers=transformers.__version__, mne=mne.__version__)
    identity['jev_config_sha256'] = {p: digest(jev_root / p) for p in ['config.json', 'model.safetensors.index.json', 'adapter_vllm/adapter_config.json', 'adapter_vllm/decision_head.json', 'adapter_vllm/adapter_model.safetensors']}
    identity['jev_shard_stats'] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(jev_root.glob('model-*.safetensors'))}
    if args.resume:
        if identity != json.loads((args.out / 'config.json').read_text()):
            raise ValueError('Resume identity changed')
    elif rank == 0:
        write_json(args.out / 'config.json', identity)
    seed_rng(CONFIG['seed'])
    print('LOADING_RELEASE_MODELS', rank, flush=True)
    text, weights, bias, tokenizer = load_jev(device=f'cuda:{device}')
    model = JEVBridge(text, weights, bias, tokenizer, True, True, head_choices=16, input_dim=256)
    model.add_module('eeg_encoder', OnlineLEAF().to(device))
    groups = [
        dict(name='connector', params=list(model.connector.parameters())),
        dict(name='jev_lora', params=[p for p in model.text.parameters() if p.requires_grad]),
        dict(name='decision_head', params=[model.head_weight, model.head_bias]),
        dict(name='eeg_encoder', params=[p for p in model.eeg_encoder.parameters() if p.requires_grad])]
    for group in groups:
        group['lr'] = CONFIG['learning_rates'][group['name']]
    active = [p for group in groups for p in group['params']]
    optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    scheduler = GroupLRSchedule(optimizer, total, 1 if args.smoke else CONFIG['warmup_steps'], CONFIG['min_lr_ratio'], freeze)
    initial = capture_model(model)
    proof = dict(trainable_sha256=tensor_fingerprint(initial['trainable']),
                 buffer_sha256=tensor_fingerprint(initial['buffers']),
                 parameter_counts={g['name']: sum(p.numel() for p in g['params']) for g in groups},
                 initialization='release LEAF + native JEV adapter + seed-specific new connector',
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    if len({p['trainable_sha256'] for p in collect(proof)}) != 1:
        raise RuntimeError('Ranks initialized differently')
    if rank == 0 and not args.resume:
        write_json(args.out / 'initial_state.json', proof)
    del initial
    buffer_proof = tensor_fingerprint(dict(model.eeg_encoder.named_buffers()))
    step, best_score, best_step = 0, -1., None
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        restore_model(model, saved)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng_by_rank'][rank])
        step, best_score, best_step = saved['step'], saved['best_score'], saved['best_step']
        del saved
    prompts = {}
    def forward(rows, training=False, variant=0):
        if not rows or len({r['split'] for r in rows}) != 1:
            raise ValueError('Empty or mixed split batch')
        if training and rows[0]['split'] != 'train':
            raise RuntimeError('Only train rows may backpropagate')
        if rows[0]['split'] == 'test':
            require_test(step, total, args.smoke)
        tid = rows[0]['task']
        if any(r['task'] != tid for r in rows):
            raise ValueError('Microbatch must have one task')
        questions = [make_variant(tasks[tid], r['label'], r['id'], step, CONFIG['seed'], None if training else variant) for r in rows]
        for q in questions:
            if q['suffix'] not in prompts:
                prompts[q['suffix']] = torch.tensor(tokenizer.encode(q['suffix'], add_special_tokens=False), device=device)
        tokens, valid = model.eeg_encoder(adapter.batch(rows), len(rows), unfrozen=training and step > freeze)
        logits = model(tokens, valid, suffix_ids=[prompts[q['suffix']] for q in questions], num_choices=len(tasks[tid]['classes']))['logits']
        labels = torch.tensor([q['label'] for q in questions], device=device)
        return logits, labels

    started = time.monotonic()
    log = (args.out / 'metrics.jsonl').open('a' if args.resume else 'x', buffering=1) if rank == 0 else None
    def report(item):
        if rank == 0:
            log.write(json.dumps(item, allow_nan=False) + '\n')
            write_json(args.out / 'progress.json', item)
            print(json.dumps(item, allow_nan=False), flush=True)

    def save(filename, include_optimizer=True):
        guard()
        if buffer_proof != tensor_fingerprint(dict(model.eeg_encoder.named_buffers())):
            raise RuntimeError('Frozen encoder buffers changed')
        states = collect(capture_rng())
        if rank == 0:
            value = dict(**capture_model(model), step=step, identity=identity,
                         best_score=best_score, best_step=best_step, rng_by_rank=states)
            if include_optimizer:
                value.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            atomic_save(value, args.out / filename)
        barrier()

    @torch.no_grad()
    def evaluate(split, cap=None, save_predictions=False):
        if split == 'test':
            require_test(step, total, args.smoke)
        model.eval()
        summary = {}
        for tid, task in enumerate(tasks):
            if task['role'] != 'benchmark':
                continue
            rows = store.evaluation(tid, split, cap)
            own = list(range(rank, len(rows), world))
            parts = []
            for offset in range(0, len(own), CONFIG['eval_microbatch']):
                ids = own[offset:offset + CONFIG['eval_microbatch']]
                logits, _ = forward([rows[i] for i in ids])
                probabilities = logits.softmax(-1).float().cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise FloatingPointError('Invalid evaluation probabilities')
                parts.extend((i, p) for i, p in zip(ids, probabilities))
            all_parts = sorted([p for part in collect(parts) for p in part], key=lambda x: x[0])
            if [i for i, _ in all_parts] != list(range(len(rows))):
                raise RuntimeError('Incomplete or duplicate evaluation rows')
            probabilities = np.stack([p for _, p in all_parts])
            labels = np.array([r['label'] for r in rows])
            metric = classification_metrics(labels, probabilities, len(task['classes']))
            subjects = np.array([r['subject'] for r in rows])
            metric['per_subject'] = {s: classification_metrics(labels[subjects == s], probabilities[subjects == s], len(task['classes'])) for s in sorted(set(subjects))}
            metric['subject_macro_balanced_accuracy'] = float(np.mean([m['balanced_accuracy'] for m in metric['per_subject'].values()]))
            summary[task['task_id']] = metric
            if rank == 0:
                print('EVALUATED', split, task['task_id'], metric['n'], metric['accuracy'], flush=True)
                if save_predictions:
                    np.savez_compressed(args.out / f'{split}_{task["task_id"]}_predictions.npz', ids=np.array([r['id'] for r in rows]), record_ids=np.array([r['record_id'] for r in rows]), subjects=subjects, labels=labels, probabilities=probabilities)
        score = float(np.mean([m['balanced_accuracy'] for m in summary.values()]))
        if rank == 0:
            write_json(args.out / f'{split}_step{step:06d}_{"subset" if cap else "full"}.json', dict(step=step, selection_checkpoint=best_step, macro_balanced_accuracy=score, datasets=summary))
        return score, summary

    try:
        # Full smoke uses two real train updates, including encoder unfreezing.
        if step == 0 and not args.smoke:
            score, _ = evaluate('valid', CONFIG['validation_per_dataset'])
            best_score, best_step = score, 0
            save('best.pt', include_optimizer=False)
            report(dict(status='validation', step=0, macro_balanced_accuracy=score, test_evaluations=0))
        for update in range(step + 1, total + 1):
            step = update
            guard()
            tid, rows = store.sample(step, batch_size, CONFIG['seed'])
            local = rows[rank::world]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rates = scheduler.set_step(step)
            loss_sum, correct = 0., 0
            for offset in range(0, len(local), CONFIG['microbatch']):
                batch = local[offset:offset + CONFIG['microbatch']]
                logits, labels = forward(batch, training=True)
                loss = F.cross_entropy(logits, labels, reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss / len(local)).backward()
                loss_sum += float(loss.detach())
                correct += int((logits.argmax(-1) == labels).sum())
            flags = torch.tensor([p.grad is not None for p in active], device=device, dtype=torch.int32)
            if world > 1:
                count = flags.clone()
                dist.all_reduce(count)
                if ((count != 0) & (count != world)).any():
                    raise RuntimeError('Ranks disagree on active gradient path')
            parameters = [p for p in active if p.grad is not None]
            encoder_grads = [p.grad for p in groups[3]['params'] if p.grad is not None]
            if bool(encoder_grads) != (step > freeze):
                raise RuntimeError('Encoder freeze/unfreeze gradient contract')
            average_gradients(parameters, world)
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            encoder_norm = float(torch.stack([g.float().square().sum() for g in encoder_grads]).sum().sqrt()) if encoder_grads else 0.
            optimizer.step()
            sums = torch.tensor([loss_sum, correct], device=device)
            if world > 1:
                dist.all_reduce(sums)
            report(dict(status='training', step=step, total_steps=total, dataset=tasks[tid]['task_id'],
                        loss=float(sums[0]) / batch_size, accuracy=float(sums[1]) / batch_size,
                        gradient_norm=float(grad), encoder_gradient_norm=encoder_norm,
                        learning_rates=rates, elapsed_seconds=time.monotonic() - started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, test_evaluations=0))
            should_stop = torch.tensor(int(stopping), device=device)
            if world > 1:
                dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
            if bool(should_stop):
                save('last.pt')
                report(dict(status='stopped', step=step, test_evaluations=0))
                return
            if not args.smoke and (step % CONFIG['validation_every'] == 0 or step == total):
                score, _ = evaluate('valid', CONFIG['validation_per_dataset'])
                if score > best_score:
                    best_score, best_step = score, step
                    save('best.pt', include_optimizer=False)
                report(dict(status='validation', step=step, macro_balanced_accuracy=score,
                            best_score=best_score, best_step=best_step, test_evaluations=0))
            if step % CONFIG['checkpoint_every'] == 0 or step == total:
                save('last.pt')
        # On-disk serialization check uses TRAIN inputs only.
        model.eval()
        rows = store.sample(1, world * 2, CONFIG['seed'])[1][rank::world]
        with torch.no_grad():
            before, _ = forward(rows)
            model.head_bias.add_(1.)
            saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            after, _ = forward(rows)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            del saved
        if rank == 0:
            write_json(args.out / 'reload_check.json', dict(logits_exact_match=True, split='train', step=step))
        if args.smoke:
            if rank == 0:
                write_json(args.out / 'complete.json', dict(smoke=True, updates=step, test_evaluations=0, checkpoint_reload_verified=True))
            return
        saved = torch.load(args.out / 'best.pt', map_location='cpu', mmap=True, weights_only=False)
        restore_model(model, saved)
        del saved
        _, valid_metrics = evaluate('valid', save_predictions=True)
        _, test_metrics = evaluate('test', save_predictions=True)
        if rank == 0:
            write_json(args.out / 'complete.json', dict(smoke=False, updates=step, selected_step=best_step,
                       selection_score=best_score, validation=valid_metrics, test=test_metrics,
                       test_evaluations=1, known_risks=CONFIG['known_risks_accepted']))
        report(dict(status='complete', step=step, selected_step=best_step, test_evaluations=1))
    finally:
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
