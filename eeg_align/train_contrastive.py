"""One GPU per fixed-text-bank objective; pure LEAF Tower adaptation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from eeg_align.benchmark_data import classification_metrics
from eeg_align.leaf_data import LEAFSignalAdapter
from eeg_align.online_leaf import OnlineLEAFTower, encode_batch
from eeg_align.checkpoint import capture_model, restore_model, seed_rng, capture_rng, restore_rng, tensor_fingerprint
from eeg_align.distributed import atomic_save
from eeg_align.evaluation import require_split, validation_rows, fixed_training_rows
from eeg_align.multitask_data import TrialStore, write_json
from eeg_align.runtime import digest
from .contrastive_protocol import CONFIG, DATA, build_text_manifest, code_identity, rates_at, load_text_bank
from .contrastive_objectives import positive_mask, per_sample_loss, class_scores


class AlignmentModel(nn.Module):
    def __init__(self, arm, text_vectors):
        super().__init__()
        self.encoder = OnlineLEAFTower(DATA)
        seed_rng(CONFIG['seed'])
        width = CONFIG['projection_width']
        self.projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, width),
                                        nn.GELU(), nn.Linear(width, width))
        self.log_scale = nn.Parameter(torch.tensor(math.log(1/CONFIG['initial_temperature'])))
        if arm == 'sigmoid':
            self.bias = nn.Parameter(torch.tensor(float(CONFIG['initial_sigmoid_bias']), dtype=torch.float32))
        else:
            self.register_buffer('bias', torch.tensor(0.))
        self.register_buffer('text_vectors', torch.as_tensor(text_vectors, dtype=torch.float32))

    def forward(self, adapted, size):
        # No label, task text, question or candidate embedding enters the EEG path.
        tokens, valid = encode_batch(self.encoder, adapted, size)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            z = self.projection(pooled)
        return F.normalize(z.float(), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=CONFIG['arms'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bank-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.cuda.set_device(0)
    if args.resume:
        if not (args.out/'last.pt').exists() or (args.out/'complete.json').exists():
            raise ValueError('Resume requires an unfinished checkpointed run')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    hashes = code_identity()
    bank, vectors = load_text_bank(args.bank_root)
    bank_hash = digest(args.bank_root/'text_bank.json')
    vector_hash = bank['vectors_sha256']
    checkpoint_stat = Path(DATA['checkpoint']).stat()
    checkpoint_stat = (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
    def guard():
        stat = Path(DATA['checkpoint']).stat()
        if (hashes != code_identity() or bank_hash != digest(args.bank_root/'text_bank.json')
                or vector_hash != digest(args.bank_root/'text_vectors.npy')
                or (stat.st_size, stat.st_mtime_ns) != checkpoint_stat):
            raise RuntimeError('Live alignment code/config/text manifest changed')
    store, adapter = TrialStore(), LEAFSignalAdapter()
    total = len(store.tasks) if args.smoke else CONFIG['steps']
    identity = dict(arm=args.arm, smoke=args.smoke, config=CONFIG, data_config=DATA,
        code_sha256=hashes, bank_sha256=bank_hash, text_vectors_sha256=bank['vectors_sha256'],
        cache_indexes={store.tasks[t]['task_id']:m['index_sha256'] for t,m in store.meta.items()},
        checkpoint_sha256=DATA['checkpoint_sha256'], torch_version=str(torch.__version__))
    if args.resume:
        if identity != json.loads((args.out/'config.json').read_text()):
            raise ValueError('Resume identity mismatch')
    else:
        write_json(args.out/'config.json', identity)
    seed_rng(CONFIG['seed'])
    model = AlignmentModel(args.arm, vectors).cuda()
    groups = [dict(name='encoder', params=[p for p in model.encoder.parameters() if p.requires_grad]),
              dict(name='projection', params=list(model.projection.parameters())),
              dict(name='scale_bias', params=[model.log_scale] + ([model.bias] if args.arm=='sigmoid' else []), weight_decay=0.)]
    for group in groups:
        group['lr'] = CONFIG['learning_rates'][group['name']]
    optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG['weight_decay'])
    active = [p for g in groups for p in g['params']]
    initial_encoder = tensor_fingerprint(model.encoder.state_dict())
    initial = dict(encoder_sha256=initial_encoder,
        buffer_sha256=tensor_fingerprint(dict(model.encoder.named_buffers())),
        projection_sha256=tensor_fingerprint(model.projection.state_dict()),
        text_sha256=tensor_fingerprint({'text':model.text_vectors}),
        parameter_counts={g['name']:sum(p.numel() for p in g['params']) for g in groups},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initialization='released LEAF Tower + new mean-pooling projection; fixed MPNet; no FiLM/Q-Former/JEV')
    if not args.resume:
        write_json(args.out/'initial_state.json', initial)
    step, best_score, best_step = 0, -1., None
    seed_rng(CONFIG['seed'])
    if args.resume:
        saved = torch.load(args.out/'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint identity mismatch')
        restore_model(model, saved)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        step, best_score, best_step = saved['step'], saved['best_score'], saved['best_step']
        del saved
    candidates = {}
    for tid, task in enumerate(store.tasks):
        item = bank['banks'][task['task_id']]
        candidates[tid] = (torch.tensor(item['text_indices'], device='cuda'),
                           torch.tensor(item['class_labels'], device='cuda'))

    def forward(rows, training=False):
        splits = {r['split'] for r in rows}
        if not rows or len(splits) != 1:
            raise ValueError('Empty or mixed split')
        require_split(rows[0]['split'])
        if training and splits != {'train'}:
            raise RuntimeError('Held-out EEG cannot backpropagate')
        tid = rows[0]['task']
        if any(r['task'] != tid for r in rows):
            raise ValueError('One candidate set per batch required')
        # Adapter reads only signals, montage, sampling rate, lengths and units.
        z = model(adapter.batch(rows), len(rows))
        indices, text_labels = candidates[tid]
        cosine = z @ model.text_vectors[indices].T
        logits = cosine * model.log_scale.exp() + model.bias
        labels = torch.tensor([int(r['label']) for r in rows], device='cuda')
        loss = per_sample_loss(logits, positive_mask(labels, text_labels), args.arm)
        scores = class_scores(cosine, text_labels, len(store.tasks[tid]['classes']))
        return loss, scores, labels, z

    started = time.monotonic()
    log = (args.out/'metrics.jsonl').open('a' if args.resume else 'x', buffering=1)
    def report(item):
        item.update(arm=args.arm, test_evaluations=0, elapsed_seconds=time.monotonic()-started)
        log.write(json.dumps(item, allow_nan=False)+'\n')
        write_json(args.out/'progress.json', item)
        print(json.dumps(item), flush=True)

    def save(name, with_optimizer=True):
        guard()
        if tensor_fingerprint(dict(model.encoder.named_buffers())) != initial['buffer_sha256']:
            raise RuntimeError('LEAF eval-mode buffers changed')
        if tensor_fingerprint({'text':model.text_vectors}) != initial['text_sha256']:
            raise RuntimeError('Frozen text bank changed')
        saved = dict(**capture_model(model), identity=identity, step=step, best_score=best_score,
                     best_step=best_step, rng=capture_rng())
        if with_optimizer:
            saved['optimizer'] = optimizer.state_dict()
        atomic_save(saved, args.out/name)

    @torch.no_grad()
    def evaluate(split, cap):
        require_split(split)
        model.eval()
        result = {}
        for tid, task in enumerate(store.tasks):
            if task['role'] != 'benchmark':
                continue
            rows = fixed_training_rows(store, tid, cap) if split=='train' else validation_rows(store, tid, cap)
            scores, losses, embeddings = [], [], []
            for offset in range(0, len(rows), CONFIG['eval_microbatch']):
                loss, prediction, _, z = forward(rows[offset:offset+CONFIG['eval_microbatch']])
                losses.extend(loss.cpu().tolist())
                scores.extend(prediction.cpu().numpy())
                if cap is None:
                    embeddings.extend(z.cpu().numpy())
            labels = np.array([int(r['label']) for r in rows])
            scores, losses = np.asarray(scores), np.asarray(losses)
            if not np.isfinite(scores).all() or not np.isfinite(losses).all():
                raise FloatingPointError('Nonfinite validation')
            metrics = classification_metrics(labels, scores, len(task['classes']))
            metrics.update(loss=float(losses.mean()), loss_kind=args.arm,
                sample_ids_sha256=hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest())
            subjects = np.array([r['subject'] for r in rows])
            metrics['per_subject'] = {s:classification_metrics(labels[subjects==s], scores[subjects==s], len(task['classes'])) for s in sorted(set(subjects))}
            result[task['task_id']] = metrics
            if cap is None:
                np.savez_compressed(args.out/f'valid_{task["task_id"]}_predictions.npz',
                    ids=np.array([r['id'] for r in rows]), subjects=subjects, labels=labels,
                    class_cosine_scores=scores, aligned_embeddings=np.asarray(embeddings))
            report(dict(status='evaluating', step=step, split=split, dataset=task['task_id'],
                        n=len(rows), accuracy=metrics['accuracy'], balanced_accuracy=metrics['balanced_accuracy'], loss=metrics['loss']))
        summary = dict(step=step, selected_step=best_step, datasets=result,
            macro_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in result.values()])),
            macro_accuracy=float(np.mean([m['accuracy'] for m in result.values()])),
            macro_loss=float(np.mean([m['loss'] for m in result.values()])))
        write_json(args.out/f'{split}_step{step:06d}_{"full" if cap is None else "subset"}.json', summary)
        return summary

    completed = step
    try:
        if not args.smoke and step == 0 and not args.resume:
            baseline = evaluate('valid', CONFIG['validation_per_dataset'])
            best_score, best_step = baseline['macro_balanced_accuracy'], 0
            evaluate('train', CONFIG['train_monitor_per_dataset'])
            save('best.pt', False)
            save('last.pt')
            report(dict(status='validation', step=0, best_score=best_score, best_step=best_step))
        for step in range(step+1, total+1):
            guard()
            tid, rows = store.sample(step, CONFIG['global_batch'], CONFIG['seed'])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rates = rates_at(step, total)
            for g in groups:
                g['lr'] = rates[g['name']]
            loss_sum, correct = 0., 0
            for offset in range(0, len(rows), CONFIG['microbatch']):
                losses, scores, labels, _ = forward(rows[offset:offset+CONFIG['microbatch']], True)
                loss = losses.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss/len(rows)).backward()
                loss_sum += float(loss.detach())
                correct += int((scores.argmax(-1)==labels).sum())
            norms = {g['name']:float(torch.stack([p.grad.float().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt()) for g in groups}
            if any(norms[k] <= 0 for k in ('encoder','projection')):
                raise RuntimeError('Missing encoder/projection learning signal')
            if model.text_vectors.grad is not None:
                raise RuntimeError('Unused/frozen parameters received gradients')
            grad = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.log_scale.clamp_(0., math.log(100.))
            completed = step
            report(dict(status='training', step=step, total_steps=total, dataset=store.tasks[tid]['task_id'],
                loss=loss_sum/len(rows), accuracy=correct/len(rows), gradient_norm=float(grad),
                group_gradient_norms=norms, learning_rates=rates, logit_scale=float(model.log_scale.exp().detach()),
                sigmoid_bias=float(model.bias.detach()),
                batch_ids_sha256=hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            if stopping:
                save('last.pt')
                report(dict(status='stopped', step=step))
                return
            if not args.smoke and (step%CONFIG['validation_every']==0 or step==total):
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
            if step%CONFIG['checkpoint_every']==0 or step==total:
                save('last.pt')
        # Always test serialization on TRAIN rows, independent of model selection.
        model.eval()
        with torch.no_grad():
            rows = store.sample(1, 2, CONFIG['seed'])[1]
            before = forward(rows)[1]
            model.projection[-1].bias.add_(1.)
            saved = torch.load(args.out/'last.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            after = forward(rows)[1]
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            del saved
        last_encoder = tensor_fingerprint(model.encoder.state_dict())
        if last_encoder == initial_encoder:
            raise RuntimeError('Encoder never changed')
        full = None
        if args.smoke:
            evaluate('train', 16)
        if not args.smoke:
            saved = torch.load(args.out/'best.pt', map_location='cpu', mmap=True, weights_only=False)
            restore_model(model, saved)
            del saved
            full = evaluate('valid', None)
        write_json(args.out/'input_audit.json', adapter.audit)
        write_json(args.out/'complete.json', dict(updates=step, smoke=args.smoke,
            selected_step=best_step, selection_score=best_score, validation=full,
            encoder_updated=True, last_encoder_sha256=last_encoder,
            selected_encoder_sha256=tensor_fingerprint(model.encoder.state_dict()),
            checkpoint_reload_verified=True, frozen_text_verified=True,
            test_evaluations=0, jev_loaded=False))
        report(dict(status='complete', step=step, selected_step=best_step))
    except BaseException as exc:
        failure = dict(status='failed', attempted_step=step, completed_training_step=completed,
                       error=f'{type(exc).__name__}: {exc}')
        write_json(args.out/'failure.json', failure)
        report(failure)
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
