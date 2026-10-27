"""LEAF -> JEV SFT on one locked official outer fold; test only at final epoch."""
import argparse
import json
import math
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .leaf_runtime import CONFIG, digest
from .benchmark_data import catalog, classification_metrics
from .official_data import OfficialFold, epoch_batches, require_final_test, import_file, write_json
from .signal_data import SignalStore
from .online_leaf import OnlineLEAF
from .task_options import make_variant
from .jev import load_jev, JEVBridge
from .distributed import atomic_save
from .checkpoint import seed_rng, capture_rng, restore_rng, capture_model, restore_model, tensor_fingerprint
from .lr_schedule import GroupLRSchedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.out / 'complete.json').exists():
        raise FileExistsError('Completed run cannot be overwritten')
    if args.resume:
        if not (args.out / 'last.pt').exists():
            raise FileNotFoundError('Resume needs this run directory/last.pt')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(3)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Worker must be restricted to exactly one assigned GPU')
    torch.cuda.set_device(0)
    fold = OfficialFold(CONFIG, args.dataset, args.seed, args.fold)
    task = catalog()[fold.task]
    # Do not use historical caches with excluded rows, merged subjects or pooled scaling.
    signals = SignalStore({args.dataset}, official_source=fold.data)
    train = fold.train
    epoch_steps = math.ceil(len(train) / CONFIG['batch_size'])
    total = 2 if args.smoke else epoch_steps * CONFIG['epochs']
    freeze = 1 if args.smoke else min(CONFIG['encoder_warmup_steps'], total // 10)
    watched = list(Path('eeg_align').glob('*.py')) + [
        Path('configs/benchmark_train.json'), Path('configs/benchmark_tasks.json'),
        Path('scripts/prepare_official_splits.py')]
    hashes = {str(p): digest(p) for p in watched}
    def guard():
        if hashes != {str(p): digest(p) for p in watched}:
            raise RuntimeError('Live code or recipe changed')

    identity = dict(dataset=args.dataset, seed=args.seed, fold=args.fold, smoke=args.smoke,
        config=CONFIG, total_steps=total, train_n=len(train), test_n=len(fold.test),
        split=fold.spec, source_sha256=fold.manifest['source_sha256'],
        code_sha256=hashes, leaf_checkpoint_sha256=digest(CONFIG['checkpoint']),
        task_sha256=digest('configs/benchmark_tasks.json'),
        instruction_sha256=digest('artifacts/benchmark/instructions.npy'))
    if args.resume:
        if json.loads((args.out / 'config.json').read_text()) != identity:
            raise ValueError('Resume identity/recipe/source mismatch')
    else:
        write_json(args.out / 'config.json', identity)
    seed_rng(args.seed)
    print('LOADING_RELEASE_WEIGHTS', args.dataset, args.seed, args.fold, flush=True)
    text, weight, bias, tokenizer = load_jev()
    model = JEVBridge(text, weight, bias, tokenizer, True, True, head_choices=16, input_dim=256)
    model.add_module('eeg_encoder', OnlineLEAF('frozen', eval_mode=True).cuda())
    initial = capture_model(model)
    initial_proof = dict(trainable_sha256=tensor_fingerprint(initial['trainable']),
        buffers_sha256=tensor_fingerprint(initial['buffers']),
        encoder_sha256=tensor_fingerprint(dict(model.eeg_encoder.named_parameters())),
        initialization='release LEAF + native JEV adapter + seed-specific fresh connector',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.resume:
        previous = json.loads((args.out / 'initial_state.json').read_text())
        for key in ('trainable_sha256', 'buffers_sha256', 'encoder_sha256'):
            if previous[key] != initial_proof[key]:
                raise ValueError('Release initialization no longer matches')
    else:
        write_json(args.out / 'initial_state.json', initial_proof)
    del initial
    bn_initial = tensor_fingerprint(dict(model.eeg_encoder.named_buffers()))
    groups = [
        dict(name='connector', params=list(model.connector.parameters()), lr=CONFIG['learning_rates'][0]),
        dict(name='jev_lora', params=[p for p in model.text.parameters() if p.requires_grad], lr=CONFIG['learning_rates'][1]),
        dict(name='decision_head', params=[model.head_weight, model.head_bias], lr=CONFIG['learning_rates'][2]),
        dict(name='eeg_encoder', params=[p for p in model.eeg_encoder.parameters() if p.requires_grad], lr=CONFIG['encoder_learning_rate'])]
    optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    active = [p for group in groups for p in group['params']]
    scheduler = GroupLRSchedule(optimizer, total,
        1 if args.smoke else math.ceil(total * CONFIG['lr_schedule']['warmup_ratio']),
        CONFIG['lr_schedule']['min_ratio'], freeze)
    step = 0
    if args.resume:
        saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Cannot resume another fold/seed/dataset')
        restore_model(model, saved)
        step = saved['step']
        if step < total:
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        del saved
    prompts = {}
    def forward(rows, variant=None, training=False):
        if any(r['split'] != ('train' if training else rows[0]['split']) for r in rows):
            raise ValueError('Mixed split batch')
        if rows[0]['split'] == 'test':
            require_final_test(step, total, args.smoke)
            if training:
                raise RuntimeError('Test loss/backprop forbidden')
        questions = [make_variant(task, r['label'], r['id'], step, args.seed, variant) for r in rows]
        for q in questions:
            if q['suffix'] not in prompts:
                prompts[q['suffix']] = torch.tensor(tokenizer.encode(q['suffix'], add_special_tokens=False), device='cuda')
        tokens, valid = model.eeg_encoder(signals.batch(rows), len(rows), unfrozen=training and step > freeze)
        logits = model(tokens, valid, suffix_ids=[prompts[q['suffix']] for q in questions],
                       num_choices=len(task['classes']))['logits']
        targets = torch.tensor([q['label'] for q in questions], device='cuda')
        return logits, targets

    def save(final=False):
        guard()
        if tensor_fingerprint(dict(model.eeg_encoder.named_buffers())) != bn_initial:
            raise RuntimeError('Frozen LEAF buffers changed')
        state = dict(**capture_model(model), step=step, identity=identity, rng=capture_rng(),
                     selection='fixed_final_epoch', training_finished=step == total)
        if not final:
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        atomic_save(state, args.out / 'last.pt')
        return state

    @torch.no_grad()
    def roundtrip(saved):
        rng = capture_rng()
        model.eval()
        rows = fold.rows(train[:CONFIG['eval_microbatch']], 'train')
        before, _ = forward(rows, variant=0)
        model.head_weight[0, 0].add_(1.)
        restore_model(model, saved)
        after, _ = forward(rows, variant=0)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        restore_rng(rng)
        proof = dict(step=step, split='train', n=len(rows), logits_exact_match=True)
        write_json(args.out / 'reload_check.json', proof)

    started = time.monotonic()
    start_step = step
    writer = SummaryWriter(str(args.out / 'tensorboard'))
    try:
        with (args.out / 'metrics.jsonl').open('a' if args.resume else 'x', buffering=1) as log:
            if step == 0:
                model.eval()
                with torch.no_grad():
                    logits, y = forward(fold.rows(train[:CONFIG['microbatch']], 'train'), variant=0)
                    value = float(F.cross_entropy(logits, y))
                log.write(json.dumps(dict(step=0, split='train_diagnostic', loss=value)) + '\n')
                writer.add_scalar('train_diagnostic/initial_loss', value, 0)
            cached_epoch = None
            batches = None
            for update in range(step + 1, total + 1):
                step = update
                epoch, offset = divmod(step - 1, epoch_steps)
                if cached_epoch != epoch:
                    batches = epoch_batches(train, args.seed, epoch, CONFIG['batch_size'])
                    cached_epoch = epoch
                indices = batches[offset]
                rows = fold.rows(indices, 'train')
                rates = scheduler.set_step(step)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum, correct = 0., 0
                for pos in range(0, len(rows), CONFIG['microbatch']):
                    batch = rows[pos:pos + CONFIG['microbatch']]
                    logits, y = forward(batch, training=True)
                    loss = F.cross_entropy(logits, y, reduction='sum')
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    (loss / len(rows)).backward()
                    loss_sum += float(loss.detach())
                    correct += int((logits.argmax(-1) == y).sum())
                encoder_grads = [p.grad for p in groups[3]['params'] if p.grad is not None]
                encoder_norm = float(torch.stack([g.float().square().sum() for g in encoder_grads]).sum().sqrt()) if encoder_grads else 0.
                if (step > freeze and not encoder_grads) or (step <= freeze and encoder_grads):
                    raise RuntimeError('Encoder freeze/unfreeze gradient contract failed')
                grad = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
                optimizer.step()
                item = dict(step=step, total_steps=total, epoch=epoch + 1, epoch_steps=epoch_steps,
                    batch_n=len(rows), loss=loss_sum / len(rows), train_accuracy=correct / len(rows),
                    learning_rates=rates, encoder_gradient_norm=encoder_norm,
                    gradient_norm=float(grad), elapsed_seconds=time.monotonic() - started,
                    resumed_from=start_step, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    test_evaluations=0, status='training')
                log.write(json.dumps(item, allow_nan=False) + '\n')
                write_json(args.out / 'progress.json', item)
                for key in ('loss', 'train_accuracy', 'encoder_gradient_norm'):
                    writer.add_scalar('train/' + key, item[key], step)
                for name, value in rates.items():
                    writer.add_scalar('learning_rate/' + name, value, step)
                if step == start_step + 1 or step % 10 == 0:
                    print('TRAIN', json.dumps(item), flush=True)
                if stopping:
                    save()
                    write_json(args.out / 'stopped.json', item)
                    return
                if step % CONFIG['checkpoint_every'] == 0 and step < total:
                    save()
            saved = save(final=True)
            # Test the actual on-disk checkpoint, never a pre-serialization copy.
            del saved
            saved = torch.load(args.out / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
            roundtrip(saved)
            del saved
            if args.smoke:
                write_json(args.out / 'complete.json', dict(smoke=True, step=step,
                    test_evaluations=0, checkpoint_reload_verified=True))
                return
            require_final_test(step, total)
            model.eval()
            results = [[], []]
            with torch.no_grad():
                for offset in range(0, len(fold.test), CONFIG['eval_microbatch']):
                    ix = fold.test[offset:offset + CONFIG['eval_microbatch']]
                    rows = fold.rows(ix, 'test')
                    for variant in (0, 1):
                        logits, _ = forward(rows, variant=variant)
                        probabilities = logits.softmax(-1).float().cpu().numpy()
                        results[variant].append(probabilities if variant == 0 else probabilities[:, ::-1])
                    write_json(args.out / 'progress.json', dict(status='final_test', step=step,
                        evaluated=min(offset + len(rows), len(fold.test)), total=len(fold.test)))
            normal, reverse = [np.concatenate(v) for v in results]
            if normal.shape != (len(fold.test), len(task['classes'])) or not np.isfinite(normal).all():
                raise RuntimeError('Incomplete/invalid final test predictions')
            y = fold.labels[fold.test]
            official = import_file('official_metrics_training', Path(CONFIG['benchmark_root']) / 'utils/metrics.py')
            metric_names = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1']
            if len(task['classes']) == 2:
                metric_names += ['auroc', 'auprc']
            report = dict(official_metrics=official.get_metrics(normal, y, metric_names, is_binary=len(task['classes']) == 2),
                normal=classification_metrics(y, normal, len(task['classes'])),
                reversed=classification_metrics(y, reverse, len(task['classes'])),
                option_agreement=float(np.mean(normal.argmax(-1) == reverse.argmax(-1))),
                step=step, epochs=CONFIG['epochs'], test_evaluations=1,
                model_selection='none; fixed final epoch', elapsed_s=time.monotonic() - started)
            np.savez_compressed(args.out / 'test_predictions.npz', source_indices=fold.test,
                subjects=fold.subjects[fold.test], labels=y, probabilities=normal, reversed_probabilities=reverse)
            report['checkpoint_sha256'] = digest(args.out / 'last.pt')
            report['predictions_sha256'] = digest(args.out / 'test_predictions.npz')
            write_json(args.out / 'complete.json', report)
            write_json(args.out / 'progress.json', dict(status='complete', step=step, test_evaluations=1))
            print('COMPLETE', json.dumps(report), flush=True)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
