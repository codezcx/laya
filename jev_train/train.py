"""Synchronous four-GPU LEAF Tower -> MLP -> MedJev supervised training."""
import argparse
from datetime import timedelta
import os
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .runtime import CONFIG, digest, arm_settings, code_identity, backbone_files
from .data import TrialStore, PreparedAdapter, write_json, classification_metrics
from .online_leaf import OnlineLEAFTower, encode_batch
from .medjev import load_medjev
from .task_options import make_variant
from .checkpoint import capture_model, restore_model, seed_rng, capture_rng, restore_rng, tensor_fingerprint
from .distributed import atomic_save, average_gradients, rank_rows
from .lr_schedule import rates_at
from .evaluation import fixed_training_rows
from .evaluation_protocol import (POLICY_PATH, POLICY, SELECTION_SPLIT, effective_config,
                                  require_split, selection_rows, EvaluationLedger)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=CONFIG['arms'], required=True)
    parser.add_argument('--backbone', default='medjev', choices=['medjev'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    language_spec = dict(kind='medjev4b', path=CONFIG['backbone_path'], pretrained_adapter=False,
                         retained_layers=23, matching_layer=32)
    torch.set_num_threads(3)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    device = int(os.environ.get('LOCAL_RANK', 0))
    if world != CONFIG['world_size'] or CONFIG['global_batch'] % world:
        raise ValueError('Launch with the configured world size and divisible global batch')
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30),
                                device_id=torch.device('cuda', device))
    def barrier():
        if world > 1:
            dist.barrier()
    def collect(value):
        if world == 1:
            return [value]
        result = [None] * world
        dist.all_gather_object(result, value)
        return result
    if rank == 0:
        if args.resume:
            if not (args.out / 'last.pt').exists() or (args.out / 'complete.json').exists():
                raise ValueError('Resume requires an unfinished checkpointed run')
        else:
            args.out.mkdir(parents=True, exist_ok=False)
    barrier()
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    setting = arm_settings(args.arm)
    train_encoder = setting['encoder_trainable']
    if setting['encoder'] != 'leaf_tower' or setting['readout'] != 'identity':
        raise ValueError('Only the text-independent LEAF Tower is supported')
    if setting['encoder_adaptation'] != 'lora' or setting['connector_adaptation'] != 'lora':
        raise ValueError('Current approach requires shared-rank Tower and connector LoRA')
    store = TrialStore()
    adapter = PreparedAdapter(store)
    tasks = store.tasks
    total = len(tasks) if args.smoke else CONFIG['steps']
    bridge_only = 1 if args.smoke else CONFIG['bridge_only_steps']
    batch_size = CONFIG['global_batch']
    hashes = code_identity()
    source_paths = [Path(CONFIG['checkpoint'])] + backbone_files()
    def source_stats():
        return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                for p in source_paths}
    fixed_source_stats = source_stats()
    def guard():
        if hashes != code_identity() or source_stats() != fixed_source_stats:
            raise RuntimeError('Live training code/config changed')
    import transformers
    identity = dict(arm=args.arm, smoke=args.smoke, config=effective_config(CONFIG),
        source_data_config=CONFIG, evaluation_policy=POLICY, world_size=world, code_sha256=hashes,
        checkpoint_sha256=digest(CONFIG['checkpoint']),
        adaptation=setting,
        cache_indexes={tasks[t]['task_id']: m['index_sha256'] for t, m in store.meta.items()},
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__),
        language_backbone=args.backbone, language_spec=language_spec,
        language_config_sha256={str(p): digest(p) for p in backbone_files()
                               if p.suffix != '.safetensors'},
        language_shard_stats={str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                             for p in backbone_files() if p.suffix == '.safetensors'})
    if args.resume:
        if identity != json.loads((args.out / 'config.json').read_text()):
            raise ValueError('Resume identity mismatch')
    elif rank == 0:
        write_json(args.out / 'config.json', identity)
    print('LOADING_RELEASE_MODELS', args.arm, flush=True)
    seed_rng(CONFIG['seed'])
    model, tokenizer = load_medjev(f'cuda:{device}')
    # Reset after release loading; the bridge must match across aligned sources.
    seed_rng(CONFIG['seed'])
    encoder = OnlineLEAFTower(CONFIG)
    transfer = None
    initial_encoder_sha256 = tensor_fingerprint(encoder.state_dict())
    encoder = encoder.cuda()
    if rank == 0:
        write_json(args.out / 'transfer_audit.json', dict(release=encoder.release_audit, alignment=transfer))
    model.add_module('eeg_encoder', encoder)
    groups = model.parameter_groups()
    if train_encoder:
        groups.append(dict(name='eeg_encoder', params=[p for p in encoder.parameters() if p.requires_grad]))
    groups = [g for g in groups if g['params']]
    for g in groups:
        g['lr'] = CONFIG['learning_rates'][g['name']]
    optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    active = [p for g in groups for p in g['params']]
    jev_params = [p for g in groups if g['name'] in {'jev_lora', 'decision_head'} for p in g['params']]
    frozen_proof = tensor_fingerprint(model.eeg_encoder.state_dict())
    buffer_proof = tensor_fingerprint(dict(encoder.named_buffers()))
    fixed_eeg_bridge = {n:p for n,p in model.named_parameters()
                       if n.startswith(('connector.', 'eeg_encoder.')) and not p.requires_grad}
    fixed_eeg_bridge_proof = tensor_fingerprint(fixed_eeg_bridge)
    if not args.resume:
        initial = dict(
            common_connector_sha256=tensor_fingerprint(model.connector.state_dict()),
            jev_adapter_head_sha256=tensor_fingerprint({n:p for n,p in model.named_parameters()
                if p.requires_grad and not n.startswith(('readout.', 'connector.', 'eeg_encoder.'))}),
            encoder_frozen=not train_encoder,
            initial_encoder_sha256=initial_encoder_sha256,
            buffer_sha256=buffer_proof, world_size=world, readout='identity',
            tower_lora_rank=CONFIG['lora_rank'], connector_lora_rank=CONFIG['lora_rank'],
            shared_lora_alpha=CONFIG['lora_alpha'], fixed_eeg_bridge_sha256=fixed_eeg_bridge_proof,
            tower_base_frozen=True, connector_base_frozen=True,
            aligned_transfer=transfer,
            parameter_counts={g['name']: sum(p.numel() for p in g['params']) for g in groups},
            output_tokens='2 * sum(ceil(segment_samples / 100)); all branches and chunks',
            initialization='Released instruction-tuned LEAF Tower + baseQwen3.5-4B -> MedJev + new MLP; no medical weights',
            language_backbone=args.backbone,
            language_audit=model.architecture_audit,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        proofs = collect(initial)
        if any(p != proofs[0] for p in proofs[1:]):
            raise RuntimeError('Rank initializations differ')
        initial['rank_initialization_verified'] = True
        if rank == 0:
            write_json(args.out / 'initial_state.json', initial)
    step, best_score, best_step = 0, -1., None
    ledger = EvaluationLedger()
    seed_rng(CONFIG['seed'] + rank)
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        restore_model(model, saved)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng_by_rank'][rank])
        step, best_score, best_step = saved['step'], saved['best_score'], saved['best_step']
        ledger = EvaluationLedger(saved['evaluation_counts'])
        del saved
    def forward(rows, training=False):
        splits = {r['split'] for r in rows}
        if len(splits) != 1 or not rows:
            raise ValueError('Empty or mixed split batch')
        require_split(rows[0]['split'], training=training)
        tid = rows[0]['task']
        if any(r['task'] != tid for r in rows):
            raise ValueError('One task per microbatch required')
        questions = [make_variant(tasks[tid], r['label'], r['id'], step, CONFIG['seed'],
                                 None if training else 0) for r in rows]
        adapted = adapter.batch(rows)
        tokens, valid = encode_batch(model.eeg_encoder, adapted, len(rows))
        logits = model(tokens, valid, prompts=questions)
        labels = torch.tensor([q['label'] for q in questions], device='cuda')
        return logits, labels
    log = (args.out / 'metrics.jsonl').open('a' if args.resume else 'x', buffering=1) if rank == 0 else None
    started = time.monotonic()
    def report(item):
        if rank != 0:
            return
        item.update(arm=args.arm, **ledger.state_dict(), elapsed_seconds=time.monotonic()-started)
        log.write(json.dumps(item, allow_nan=False) + '\n')
        write_json(args.out / 'progress.json', item)
        print(json.dumps(item), flush=True)

    def save(filename, optimizer_state=True):
        guard()
        if tensor_fingerprint(dict(encoder.named_buffers())) != buffer_proof:
            raise RuntimeError('Frozen LEAF buffers changed')
        if tensor_fingerprint(fixed_eeg_bridge) != fixed_eeg_bridge_proof:
            raise RuntimeError('Frozen Tower/connector base changed during LoRA training')
        if (not train_encoder or step <= bridge_only) and frozen_proof != tensor_fingerprint(model.eeg_encoder.state_dict()):
            raise RuntimeError('Frozen encoder weights/buffers changed')
        if not all(p.requires_grad for p in active):
            raise RuntimeError('Snapshot must include both training phases')
        rngs = collect(capture_rng())
        fingerprints = collect(tensor_fingerprint({n:p for n,p in model.named_parameters() if p.requires_grad}))
        if len(set(fingerprints)) != 1:
            raise RuntimeError('Synchronized ranks have diverged')
        if rank == 0:
            value = dict(**capture_model(model), identity=identity, step=step, best_score=best_score,
                         best_step=best_step, rng_by_rank=rngs, rank_weights_verified=True,
                         evaluation_counts=ledger.state_dict())
            if optimizer_state:
                value['optimizer'] = optimizer.state_dict()
            atomic_save(value, args.out / filename)
        barrier()

    @torch.no_grad()
    def evaluate(split, cap):
        require_split(split)
        if split == 'train' and not cap:
            raise ValueError('Train monitoring requires a cap')
        model.eval()
        summary, manifest = {}, {}
        selection_ids_path = args.out / f'{split}_selection_ids.json'
        prior_ids = (json.loads(selection_ids_path.read_text())
                     if split == SELECTION_SPLIT and cap is not None and selection_ids_path.exists() else None)
        for tid, task in enumerate(tasks):
            if task['role'] != 'benchmark':
                continue
            rows = fixed_training_rows(store, tid, cap) if split == 'train' else selection_rows(store, tid, cap)
            manifest[task['task_id']] = [r['id'] for r in rows]
            if prior_ids is not None and prior_ids['datasets'][task['task_id']] != manifest[task['task_id']]:
                raise RuntimeError('Fixed development selection IDs changed')
            indices = list(range(rank, len(rows), world))
            local = []
            for offset in range(0, len(indices), CONFIG['eval_microbatch']):
                ix = indices[offset:offset+CONFIG['eval_microbatch']]
                logits, labels = forward([rows[i] for i in ix])
                probs = logits.softmax(-1).float().cpu().tolist()
                losses = F.cross_entropy(logits, labels, reduction='none').float().cpu().tolist()
                local.extend(zip(ix, probs, losses))
            records = sorted(item for part in collect(local) for item in part)
            if [r[0] for r in records] != list(range(len(rows))):
                raise RuntimeError('Evaluation sharding duplicated or lost samples')
            probabilities = np.array([r[1] for r in records])
            losses = np.array([r[2] for r in records])
            if not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
                raise FloatingPointError('Invalid evaluation')
            ledger.record(split, len(rows))
            labels = np.array([r['label'] for r in rows])
            metric = classification_metrics(labels, probabilities, len(task['classes']))
            metric.update(loss=float(losses.mean()), sample_ids_sha256=hashlib.sha256(
                json.dumps([r['id'] for r in rows]).encode()).hexdigest())
            subjects = np.array([r['subject'] for r in rows])
            metric['per_subject'] = {s:classification_metrics(labels[subjects==s], probabilities[subjects==s],
                                      len(task['classes'])) for s in sorted(set(subjects))}
            summary[task['task_id']] = metric
            if cap is None and rank == 0:
                np.savez_compressed(args.out / f'{split}_{task["task_id"]}_predictions.npz',
                    ids=np.array([r['id'] for r in rows]), subjects=subjects, labels=labels, probabilities=probabilities)
            report(dict(status='evaluating', step=step, split=split, dataset=task['task_id'],
                        n=len(rows), accuracy=metric['accuracy'], loss=metric['loss']))
        result = dict(step=step, selected_step=best_step, datasets=summary,
            split=split, role=POLICY['selection_role'] if split == SELECTION_SPLIT else 'train_monitor',
            scope='full' if cap is None else 'subset', cap_per_dataset=cap,
            independent_test_estimate=False, evaluation_counts=ledger.state_dict(),
            macro_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in summary.values()])),
            macro_accuracy=float(np.mean([m['accuracy'] for m in summary.values()])),
            macro_loss=float(np.mean([m['loss'] for m in summary.values()])))
        if rank == 0:
            write_json(args.out / f'{split}_step{step:06d}_{"subset" if cap else "full"}.json', result)
            if split == SELECTION_SPLIT and cap is not None and prior_ids is None:
                write_json(selection_ids_path, dict(split=split, role=POLICY['selection_role'],
                    cap_per_dataset=cap, datasets=manifest))
        barrier()
        return result

    completed_step = step
    try:
        if not args.smoke and step == 0 and not args.resume:
            baseline = evaluate(SELECTION_SPLIT, CONFIG['validation_per_dataset'])
            best_score, best_step = baseline['macro_balanced_accuracy'], 0
            evaluate('train', CONFIG['train_monitor_per_dataset'])
            save('best.pt', False)
            save('last.pt')
            report(dict(status='development', **{k:v for k,v in baseline.items() if k != 'datasets'}))
        for step in range(step+1, total+1):
            guard()
            tid, rows = store.sample(step, batch_size, CONFIG['seed'])
            batch_sha256 = hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest()
            global_ids = [r['id'] for r in rows]
            rows = rank_rows(rows, rank, world)
            if [i for part in collect([r['id'] for r in rows]) for i in part] != global_ids:
                raise RuntimeError('Distributed batch differs from planned global batch')
            if train_encoder:
                model.eeg_encoder.set_adaptation_enabled(step > bridge_only)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for p in jev_params:
                p.requires_grad_(step > bridge_only)
            rates = rates_at(step, total, bridge_only, CONFIG['learning_rates'], CONFIG['warmup_steps'], CONFIG['min_lr_ratio'])
            rates = {g['name']: rates[g['name']] for g in optimizer.param_groups}
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
            if any(p.grad is not None for p in model.eeg_encoder.parameters()) != (train_encoder and step > bridge_only):
                raise RuntimeError('Encoder phase gradient contract failed')
            if any(p.grad is not None for p in jev_params) != (step > bridge_only):
                raise RuntimeError('JEV phase gradient contract failed')
            if any(p.grad is not None or p.requires_grad for p in fixed_eeg_bridge.values()):
                raise RuntimeError('Tower/connector base must remain frozen in both stages')
            average_gradients([p for p in active if p.requires_grad], world)
            stats = torch.tensor([loss_sum, correct], dtype=torch.float64, device='cuda')
            if world > 1:
                dist.all_reduce(stats)
            loss_sum, correct = stats.tolist()
            norms = {g['name']: float(torch.stack([p.grad.float().square().sum() for p in g['params']
                     if p.grad is not None]).sum().sqrt()) if any(p.grad is not None for p in g['params']) else 0. for g in groups}
            if norms['connector'] <= 0 or ('readout' in norms and norms['readout'] <= 0):
                raise RuntimeError('Missing bridge/readout learning signal')
            if train_encoder and step > bridge_only and norms['eeg_encoder'] <= 0:
                raise RuntimeError('Missing encoder learning signal')
            grad = torch.nn.utils.clip_grad_norm_([p for p in active if p.grad is not None], 1., error_if_nonfinite=True)
            optimizer.step()
            for p in jev_params:
                p.requires_grad_(True)
            if train_encoder:
                model.eeg_encoder.set_adaptation_enabled(True)
            completed_step = step
            report(dict(status='training', step=step, total_steps=total, dataset=tasks[tid]['task_id'],
                loss=loss_sum / batch_size, accuracy=correct / batch_size, gradient_norm=float(grad),
                group_gradient_norms=norms, learning_rates=rates,
                world_size=world, local_batch=len(rows), distributed_sampling_verified=True,
                batch_ids_sha256=batch_sha256,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            stop_flag = torch.tensor(int(stopping), device='cuda')
            if world > 1:
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if stop_flag.item():
                save('last.pt')
                report(dict(status='stopped', step=step))
                return
            if not args.smoke and (step % CONFIG['validation_every'] == 0 or step == total):
                rng = capture_rng()
                result = evaluate(SELECTION_SPLIT, CONFIG['validation_per_dataset'])
                if result['macro_balanced_accuracy'] > best_score:
                    best_score, best_step = result['macro_balanced_accuracy'], step
                    save('best.pt', False)
                evaluate('train', CONFIG['train_monitor_per_dataset'])
                restore_rng(rng)
                save(f'step{step:06d}.pt', False)
                report(dict(status='development', split=SELECTION_SPLIT, role=POLICY['selection_role'],
                            step=step, macro_balanced_accuracy=result['macro_balanced_accuracy'],
                            best_score=best_score, best_step=best_step))
            if step % CONFIG['checkpoint_every'] == 0 or step in {bridge_only, bridge_only+1, total}:
                save('last.pt')
        write_json(args.out / f'input_audit_rank{rank}.json', adapter.audit)
        # Real train-only serialization test; not a meta-device shape check.
        model.eval()
        with torch.no_grad():
            rows = store.sample(1, 2, CONFIG['seed'])[1]
            before, _ = forward(rows)
            model.readout.head_mlp[-1].bias.add_(1.)
            saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            after, _ = forward(rows)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            del saved
        barrier()
        if rank == 0:
            write_json(args.out / 'reload_check.json', dict(logits_exact_match=True, split='train', verified_ranks=world))
        full = None
        if args.smoke:
            evaluate('train', 16)
        if not args.smoke:
            saved = torch.load(args.out / 'best.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            del saved
            full = evaluate(SELECTION_SPLIT, None)
        final_encoder_proof = tensor_fingerprint(model.eeg_encoder.state_dict())
        if (final_encoder_proof != frozen_proof) != (train_encoder and (args.smoke or best_step > bridge_only)):
            raise RuntimeError('Final encoder update/freeze proof failed')
        if len(set(collect(final_encoder_proof))) != 1:
            raise RuntimeError('Final encoder differs across ranks')
        if rank == 0:
            write_json(args.out / 'complete.json', dict(smoke=args.smoke, updates=step,
                selected_step=best_step, selection_score=best_score, development_evaluation=full,
                evaluation_policy=POLICY, evaluation_counts=ledger.state_dict(),
                test_evaluations=ledger.counts['test_evaluations'], valid_evaluations=ledger.counts['valid_evaluations'],
                checkpoint_reload_verified=True, encoder_update_contract_verified=True,
                encoder_frozen=not train_encoder, encoder_changed=final_encoder_proof != frozen_proof,
                frozen_eeg_bridge_verified=tensor_fingerprint(fixed_eeg_bridge) == fixed_eeg_bridge_proof,
                tower_lora_rank=CONFIG['lora_rank'], connector_lora_rank=CONFIG['lora_rank'],
                world_size=world, rank_weights_verified=True, distributed_sampling_verified=True,
                selected_encoder_sha256=final_encoder_proof))
        report(dict(status='complete', step=step, selected_step=best_step))
    except BaseException as exc:
        failure = dict(status='failed', attempted_step=step, completed_training_step=completed_step,
                       error=f'{type(exc).__name__}: {exc}')
        write_json(args.out / f'failure_rank{rank}.json', failure)
        report(failure)
        raise
    finally:
        if log is not None:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
