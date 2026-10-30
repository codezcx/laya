"""One GPU per matched BSR/JEV readout; no test access, resumable pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from .runtime import CONFIG, digest
from .multitask_data import TrialStore, write_json
from .benchmark_data import classification_metrics
from .bsr_data import BSRSignalAdapter, encode_batch
from .online_bsr import OnlineBSR, upstream_files
from .token_resampler import TokenReadout, KINDS
from .jev import JEVBridge, load_jev
from .task_options import make_variant
from .checkpoint import capture_model, restore_model, seed_rng, capture_rng, restore_rng, tensor_fingerprint
from .distributed import atomic_save
from .lr_schedule import rates_at
from .evaluation import require_split, fixed_training_rows, validation_rows


def code_identity():
    paths = sorted(Path('eeg_align').glob('*.py')) + [Path(p) for p in (
        'configs/benchmark_train.json', 'configs/benchmark_tasks.json',
        'scripts/prepare_multitask_signals.py', 'scripts/audit_multitask_signals.py')]
    paths += upstream_files(CONFIG)
    return {str(p): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=KINDS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.cuda.set_device(0)
    if args.resume:
        if not (args.out / 'last.pt').exists() or (args.out / 'complete.json').exists():
            raise ValueError('Resume requires an unfinished checkpointed run')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    store, adapter = TrialStore(), BSRSignalAdapter()
    tasks = store.tasks
    total = len(tasks) if args.smoke else CONFIG['steps']
    bridge_only = 1 if args.smoke else CONFIG['bridge_only_steps']
    batch_size = CONFIG['global_batch']
    hashes = code_identity()
    def guard():
        if hashes != code_identity():
            raise RuntimeError('Live training code/config changed')
    import transformers
    root = Path('laya/JEV-27B-VL')
    identity = dict(arm=args.arm, smoke=args.smoke, config=CONFIG, code_sha256=hashes,
        checkpoint_sha256=digest(CONFIG['checkpoint']),
        cache_indexes={tasks[t]['task_id']: m['index_sha256'] for t, m in store.meta.items()},
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__),
        jev_config_sha256={p: digest(root / p) for p in ('config.json', 'model.safetensors.index.json',
            'adapter_vllm/adapter_config.json', 'adapter_vllm/decision_head.json', 'adapter_vllm/adapter_model.safetensors')},
        jev_shard_stats={p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(root.glob('model-*.safetensors'))})
    if args.resume:
        if identity != json.loads((args.out / 'config.json').read_text()):
            raise ValueError('Resume identity mismatch')
    else:
        write_json(args.out / 'config.json', identity)
    print('LOADING_RELEASE_MODELS', args.arm, flush=True)
    seed_rng(CONFIG['seed'])
    text, weight, bias, tokenizer = load_jev('cuda:0')
    # Reset after release loading: identical shared connector across all arms.
    seed_rng(CONFIG['seed'])
    model = JEVBridge(text, weight, bias, tokenizer, True, True, head_choices=16, input_dim=768)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(CONFIG['seed'] + 100)
        model.add_module('readout', TokenReadout(args.arm).cuda())
    model.add_module('eeg_encoder', OnlineBSR(CONFIG).cuda())
    groups = [dict(name='connector', params=list(model.connector.parameters())),
        dict(name='readout', params=list(model.readout.parameters())),
        dict(name='jev_lora', params=[p for p in model.text.parameters() if p.requires_grad]),
        dict(name='decision_head', params=[model.head_weight, model.head_bias])]
    groups = [g for g in groups if g['params']]
    for g in groups:
        g['lr'] = CONFIG['learning_rates'][g['name']]
    optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    active = [p for g in groups for p in g['params']]
    jev_params = [p for g in groups if g['name'] in {'jev_lora', 'decision_head'} for p in g['params']]
    frozen_proof = tensor_fingerprint(model.eeg_encoder.state_dict())
    if not args.resume:
        write_json(args.out / 'initial_state.json', dict(
            common_connector_sha256=tensor_fingerprint(model.connector.state_dict()),
            jev_adapter_head_sha256=tensor_fingerprint({n:p for n,p in model.named_parameters()
                if p.requires_grad and not n.startswith(('readout.', 'connector.'))}),
            frozen_bsr_sha256=frozen_proof,
            parameter_counts={g['name']: sum(p.numel() for p in g['params']) for g in groups},
            output_tokens=model.readout.output_tokens,
            initialization='released BSR + native JEV + new readout/connector; no old downstream weights',
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    step, best_score, best_step = 0, -1., None
    seed_rng(CONFIG['seed'])
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        restore_model(model, saved)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        step, best_score, best_step = saved['step'], saved['best_score'], saved['best_step']
        del saved
    prompts = {}
    def forward(rows, training=False):
        splits = {r['split'] for r in rows}
        if len(splits) != 1 or not rows:
            raise ValueError('Empty or mixed split batch')
        require_split(rows[0]['split'])
        if training and splits != {'train'}:
            raise RuntimeError('Only train rows can backpropagate')
        tid = rows[0]['task']
        if any(r['task'] != tid for r in rows):
            raise ValueError('One task per microbatch required')
        questions = [make_variant(tasks[tid], r['label'], r['id'], step, CONFIG['seed'],
                                 None if training else 0) for r in rows]
        for q in questions:
            if q['suffix'] not in prompts:
                prompts[q['suffix']] = torch.tensor(tokenizer.encode(q['suffix'], add_special_tokens=False), device='cuda')
        tokens, valid = encode_batch(model.eeg_encoder, adapter.batch(rows), len(rows))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens, valid = model.readout(tokens, valid)
        logits = model(tokens, valid, suffix_ids=[prompts[q['suffix']] for q in questions],
                       num_choices=len(tasks[tid]['classes']))['logits']
        labels = torch.tensor([q['label'] for q in questions], device='cuda')
        return logits, labels
    log = (args.out / 'metrics.jsonl').open('a' if args.resume else 'x', buffering=1)
    started = time.monotonic()
    def report(item):
        item.update(arm=args.arm, test_evaluations=0, elapsed_seconds=time.monotonic()-started)
        log.write(json.dumps(item, allow_nan=False) + '\n')
        write_json(args.out / 'progress.json', item)
        print(json.dumps(item), flush=True)

    def save(filename, optimizer_state=True):
        guard()
        if frozen_proof != tensor_fingerprint(model.eeg_encoder.state_dict()):
            raise RuntimeError('Frozen BSR weights/buffers changed')
        if not all(p.requires_grad for p in active):
            raise RuntimeError('Snapshot must include both training phases')
        value = dict(**capture_model(model), identity=identity, step=step, best_score=best_score,
                     best_step=best_step, rng=capture_rng())
        if optimizer_state:
            value['optimizer'] = optimizer.state_dict()
        atomic_save(value, args.out / filename)

    @torch.no_grad()
    def evaluate(split, cap):
        require_split(split)
        if split == 'train' and not cap:
            raise ValueError('Train monitoring requires a cap')
        model.eval()
        summary, legacy = {}, {}
        for tid, task in enumerate(tasks):
            if task['role'] != 'benchmark':
                continue
            rows = fixed_training_rows(store, tid, cap) if split == 'train' else validation_rows(store, tid, cap)
            probabilities, losses = [], []
            for offset in range(0, len(rows), CONFIG['eval_microbatch']):
                logits, labels = forward(rows[offset:offset+CONFIG['eval_microbatch']])
                probabilities.extend(logits.softmax(-1).float().cpu().numpy())
                losses.extend(F.cross_entropy(logits, labels, reduction='none').float().cpu().tolist())
            probabilities, losses = np.array(probabilities), np.array(losses)
            if not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
                raise FloatingPointError('Invalid evaluation')
            labels = np.array([r['label'] for r in rows])
            metric = classification_metrics(labels, probabilities, len(task['classes']))
            metric.update(loss=float(losses.mean()), sample_ids_sha256=hashlib.sha256(
                json.dumps([r['id'] for r in rows]).encode()).hexdigest())
            subjects = np.array([r['subject'] for r in rows])
            metric['per_subject'] = {s:classification_metrics(labels[subjects==s], probabilities[subjects==s],
                                      len(task['classes'])) for s in sorted(set(subjects))}
            summary[task['task_id']] = metric
            if split == 'valid':
                ids = {r['id'] for r in store.evaluation(tid, 'valid', 128)}
                mask = np.array([r['id'] in ids for r in rows])
                legacy[task['task_id']] = classification_metrics(labels[mask], probabilities[mask], len(task['classes']))
            if cap is None:
                np.savez_compressed(args.out / f'valid_{task["task_id"]}_predictions.npz',
                    ids=np.array([r['id'] for r in rows]), subjects=subjects, labels=labels, probabilities=probabilities)
            report(dict(status='evaluating', step=step, split=split, dataset=task['task_id'],
                        n=len(rows), accuracy=metric['accuracy'], loss=metric['loss']))
        result = dict(step=step, selected_step=best_step, datasets=summary,
            macro_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in summary.values()])),
            macro_accuracy=float(np.mean([m['accuracy'] for m in summary.values()])),
            macro_loss=float(np.mean([m['loss'] for m in summary.values()])), legacy128=legacy,
            legacy128_macro_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in legacy.values()])) if legacy else None)
        write_json(args.out / f'{split}_step{step:06d}_{"subset" if cap else "full"}.json', result)
        return result

    completed_step = step
    try:
        if not args.smoke and step == 0 and not args.resume:
            baseline = evaluate('valid', CONFIG['validation_per_dataset'])
            best_score, best_step = baseline['macro_balanced_accuracy'], 0
            evaluate('train', CONFIG['train_monitor_per_dataset'])
            save('best.pt', False)
            save('last.pt')
            report(dict(status='validation', **{k:v for k,v in baseline.items() if k != 'datasets'}))
        for step in range(step+1, total+1):
            guard()
            tid, rows = store.sample(step, batch_size, CONFIG['seed'])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for p in jev_params:
                p.requires_grad_(step > bridge_only)
            rates = rates_at(step, total, bridge_only, CONFIG['learning_rates'], CONFIG['warmup_steps'], CONFIG['min_lr_ratio'])
            for g in optimizer.param_groups:
                g['lr'] = rates[g['name']]
            loss_sum, correct = 0., 0
            for offset in range(0, len(rows), CONFIG['microbatch']):
                logits, labels = forward(rows[offset:offset+CONFIG['microbatch']], training=True)
                loss = F.cross_entropy(logits, labels, reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss / len(rows)).backward()
                loss_sum += float(loss.detach())
                correct += int((logits.argmax(-1) == labels).sum())
            if any(p.grad is not None for p in model.eeg_encoder.parameters()):
                raise RuntimeError('BSR gradient leak')
            if any(p.grad is not None for p in jev_params) != (step > bridge_only):
                raise RuntimeError('JEV phase gradient contract failed')
            norms = {g['name']: float(torch.stack([p.grad.float().square().sum() for p in g['params']
                     if p.grad is not None]).sum().sqrt()) if any(p.grad is not None for p in g['params']) else 0. for g in groups}
            if norms['connector'] <= 0 or (args.arm != 'mean_pool' and norms['readout'] <= 0):
                raise RuntimeError('Missing bridge/readout learning signal')
            grad = torch.nn.utils.clip_grad_norm_([p for p in active if p.grad is not None], 1., error_if_nonfinite=True)
            optimizer.step()
            for p in jev_params:
                p.requires_grad_(True)
            completed_step = step
            report(dict(status='training', step=step, total_steps=total, dataset=tasks[tid]['task_id'],
                loss=loss_sum / batch_size, accuracy=correct / batch_size, gradient_norm=float(grad),
                group_gradient_norms=norms, learning_rates=rates,
                batch_ids_sha256=hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            if stopping:
                save('last.pt')
                report(dict(status='stopped', step=step))
                return
            if not args.smoke and (step % CONFIG['validation_every'] == 0 or step == total):
                rng = capture_rng()
                result = evaluate('valid', CONFIG['validation_per_dataset'])
                if result['macro_balanced_accuracy'] > best_score:
                    best_score, best_step = result['macro_balanced_accuracy'], step
                    save('best.pt', False)
                evaluate('train', CONFIG['train_monitor_per_dataset'])
                restore_rng(rng)
                save(f'step{step:06d}.pt', False)
                report(dict(status='validation', step=step, macro_balanced_accuracy=result['macro_balanced_accuracy'],
                            best_score=best_score, best_step=best_step))
            if step % CONFIG['checkpoint_every'] == 0 or step in {bridge_only, bridge_only+1, total}:
                save('last.pt')
        write_json(args.out / 'input_audit.json', adapter.audit)
        # Real train-only serialization test; not a meta-device shape check.
        model.eval()
        with torch.no_grad():
            rows = store.sample(1, 2, CONFIG['seed'])[1]
            before, _ = forward(rows)
            model.head_bias.add_(1.)
            saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            after, _ = forward(rows)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            del saved
        write_json(args.out / 'reload_check.json', dict(logits_exact_match=True, split='train'))
        full = None
        if not args.smoke:
            saved = torch.load(args.out / 'best.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            del saved
            full = evaluate('valid', None)
        write_json(args.out / 'complete.json', dict(smoke=args.smoke, updates=step,
            selected_step=best_step, selection_score=best_score, validation=full,
            test_evaluations=0, checkpoint_reload_verified=True, frozen_bsr_verified=True))
        report(dict(status='complete', step=step, selected_step=best_step))
    except BaseException as exc:
        failure = dict(status='failed', attempted_step=step, completed_training_step=completed_step,
                       error=f'{type(exc).__name__}: {exc}')
        write_json(args.out / 'failure.json', failure)
        report(failure)
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
