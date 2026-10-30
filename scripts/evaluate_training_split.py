"""Resume-safe full TRAIN evaluation of the selected LEAF/JEV checkpoint.

Uses the frozen prepared windows and the same canonical questions as final test
evaluation. No optimizer, resampling, split changes, or held-out forward passes.
"""
import argparse
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def training_rows(store, tid):
    rows = [r for i, r in enumerate(store.rows[tid])
            if r['split'] == 'train' and i not in store.excluded[tid]]
    name = store.tasks[tid]['task_id']
    expected = store.quality[name]['effective_counts']['train']
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError(f'Incomplete or duplicate training population: {name}')
    if store.tasks[tid]['role'] != 'benchmark':
        raise ValueError('This evaluation is restricted to the nine benchmark tasks')
    return rows


def checked_predictions(path, indices, rows, classes):
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['indices'], indices):
            raise ValueError('Prediction indices changed')
        if not np.array_equal(data['ids'], [rows[i]['id'] for i in indices]):
            raise ValueError('Prediction sample identities changed')
        if not np.array_equal(data['labels'], [rows[i]['label'] for i in indices]):
            raise ValueError('Prediction labels changed')
        probabilities = data['probabilities'].copy()
    if (probabilities.shape != (len(indices), classes)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError('Invalid saved probabilities')
    return probabilities


def save_predictions(path, indices, rows, probabilities):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, indices=np.asarray(indices, dtype=np.int64),
            ids=np.asarray([rows[i]['id'] for i in indices]),
            labels=np.asarray([rows[i]['label'] for i in indices], dtype=np.int64),
            probabilities=np.asarray(probabilities, dtype=np.float32))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def worker(args):
    import torch
    import torch.distributed as dist
    import transformers
    import mne
    from eeg_align.runtime import CONFIG, digest
    from eeg_align.benchmark_data import classification_metrics
    from eeg_align.multitask_data import TrialStore, SignalAdapter, write_json
    from eeg_align.online_leaf import OnlineLEAF
    from eeg_align.jev import load_jev, JEVBridge
    from eeg_align.checkpoint import restore_model, seed_rng, tensor_fingerprint
    from eeg_align.task_options import make_variant

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(3)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2),
            device_id=torch.device('cuda', device))

    def barrier():
        if world > 1:
            dist.barrier()

    started = time.monotonic()
    def report(stage, **values):
        item = dict(status=stage, rank=rank, split='train',
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            elapsed_seconds=round(time.monotonic() - started, 2),
            test_evaluations=0, optimizer_steps=0, **values)
        write_json(args.out / f'progress_rank{rank}.json', item)
        if rank == 0:
            write_json(args.out / 'progress.json', item)
        print(json.dumps(item), flush=True)

    try:
        original = json.loads((args.training_run / 'config.json').read_text())
        complete = json.loads((args.training_run / 'complete.json').read_text())
        runtime = dict(torch=torch.__version__, transformers=transformers.__version__, mne=mne.__version__)
        if original['config'] != CONFIG or original['runtime'] != runtime:
            raise ValueError('Recipe or runtime differs from the completed experiment')
        if complete['smoke'] or complete['updates'] != CONFIG['steps']:
            raise ValueError('Evaluation requires a completed full training run')
        watch = dict(original['code_sha256'])
        watch[str(Path(__file__).relative_to(ROOT))] = digest(__file__)
        checkpoint = args.training_run / 'best.pt'
        checkpoint_stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        def guard():
            bad = [p for p, h in watch.items() if digest(p) != h]
            if bad or checkpoint_stat != (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns):
                raise RuntimeError(f'Source or checkpoint changed: {bad}')
        guard()
        if digest(CONFIG['checkpoint']) != original['checkpoint_sha256']:
            raise ValueError('Released LEAF checkpoint changed')
        if digest(Path(CONFIG['text_root']) / 'instructions.npy') != original['instructions_sha256']:
            raise ValueError('Task instruction embeddings changed')
        jev = Path('laya/JEV-27B-VL')
        if any(digest(jev / p) != h for p, h in original['jev_config_sha256'].items()):
            raise ValueError('JEV configuration or native adapter changed')
        if original['jev_shard_stats'] != {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                for p in sorted(jev.glob('model-*.safetensors'))}:
            raise ValueError('JEV base weight files changed')
        report('checking_frozen_training_inputs')
        store = TrialStore()
        if original['cache_indexes'] != {t['task_id']: store.meta[i]['index_sha256']
                for i, t in enumerate(store.tasks)}:
            raise ValueError('Prepared data indexes differ from original evaluation')
        populations = {i: training_rows(store, i) for i, t in enumerate(store.tasks)
                       if t['role'] == 'benchmark'}
        counts = {store.tasks[i]['task_id']: len(r) for i, r in populations.items()}
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if state['identity'] != original or state['step'] != complete['selected_step']:
            raise ValueError('Checkpoint is not the selected original best model')
        identity = dict(schema='full-training-evaluation-v1', split='train',
            training_run=str(args.training_run), selected_step=state['step'],
            checkpoint_sha256=digest(checkpoint), original_identity=original,
            evaluator_sha256=digest(__file__), world_size=world,
            microbatch=args.microbatch, chunk_size=args.chunk_size,
            prompt_variant=0, expected_counts=counts,
            total_samples=sum(counts.values()), test_evaluations=0, optimizer_steps=0)
        identity_path = args.out / 'identity.json'
        if rank == 0:
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError('Existing evaluation belongs to another model or configuration')
            else:
                write_json(identity_path, identity)
        barrier()
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Ranks disagree on evaluation identity')
        seed_rng(CONFIG['seed'])
        adapter = SignalAdapter()
        report('loading_selected_model', selected_step=state['step'], expected_counts=counts)
        text, weights, bias, tokenizer = load_jev(device=f'cuda:{device}')
        model = JEVBridge(text, weights, bias, tokenizer, True, True, head_choices=16, input_dim=256)
        model.add_module('eeg_encoder', OnlineLEAF().to(device))
        restore_model(model, state)
        names = set(state['trainable'])
        expected_parameters = tensor_fingerprint(state['trainable'])
        expected_buffers = tensor_fingerprint(state['buffers'])
        del state
        model.eval().requires_grad_(False)
        def unchanged():
            if (tensor_fingerprint({n: p for n, p in model.named_parameters() if n in names}) != expected_parameters
                    or tensor_fingerprint(dict(model.named_buffers())) != expected_buffers):
                raise RuntimeError('Checkpoint parameters or buffers changed during evaluation')
        unchanged()
        prompts = {}
        @torch.no_grad()
        def forward(rows):
            if not rows or any(r['split'] != 'train' for r in rows):
                raise ValueError('Only training signals are allowed')
            tid = rows[0]['task']
            if any(r['task'] != tid for r in rows):
                raise ValueError('Mixed-task batch')
            task = store.tasks[tid]
            questions = [make_variant(task, r['label'], r['id'], identity['selected_step'], CONFIG['seed'], 0)
                         for r in rows]
            for question, row in zip(questions, rows):
                if question['label'] != row['label']:
                    raise ValueError('Canonical option order changed')
                if question['suffix'] not in prompts:
                    prompts[question['suffix']] = torch.tensor(
                        tokenizer.encode(question['suffix'], add_special_tokens=False), device=device)
            tokens, valid = model.eeg_encoder(adapter.batch(rows), len(rows), unfrozen=False)
            logits = model(tokens, valid, suffix_ids=[prompts[q['suffix']] for q in questions],
                           num_choices=len(task['classes']))['logits']
            probabilities = logits.softmax(-1).float().cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise FloatingPointError('Nonfinite inference probabilities')
            return probabilities

        report('train_only_preflight')
        for tid, rows in populations.items():
            sample = rows[rank::world][:2]
            first, second = forward(sample), forward(sample)
            np.testing.assert_allclose(first, second, rtol=0, atol=0)
        unchanged()
        write_json(args.out / f'preflight_rank{rank}.json', dict(
            datasets=len(populations), repeat_predictions_identical=True,
            checkpoint_parameters_and_buffers_unchanged=True, test_evaluations=0))
        barrier()
        summaries = {}
        # Finish small datasets early; inference order does not change the split.
        for tid in sorted(populations, key=lambda i: len(populations[i])):
            rows = populations[tid]
            task = store.tasks[tid]
            name, classes = task['task_id'], len(task['classes'])
            own = list(range(rank, len(rows), world))
            for offset in range(0, len(own), args.chunk_size):
                guard()
                indices = own[offset:offset + args.chunk_size]
                path = args.out / 'chunks' / name / f'rank{rank}_offset{offset:07d}.npz'
                metadata = path.with_suffix('.json')
                resumed = path.exists() and metadata.exists()
                if resumed:
                    if json.loads(metadata.read_text())['sha256'] != digest(path):
                        raise ValueError('Saved prediction chunk changed')
                    checked_predictions(path, indices, rows, classes)
                else:
                    parts = [forward([rows[i] for i in indices[start:start + args.microbatch]])
                             for start in range(0, len(indices), args.microbatch)]
                    save_predictions(path, indices, rows, np.concatenate(parts))
                    checked_predictions(path, indices, rows, classes)
                    write_json(metadata, dict(sha256=digest(path), n=len(indices)))
                report('evaluating_train', dataset=name, rank_processed=offset + len(indices),
                    rank_total=len(own), dataset_total=len(rows), resumed_chunk=resumed,
                    completed_datasets=list(summaries), peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            barrier()
            if rank == 0:
                probabilities = np.empty((len(rows), classes), dtype=np.float32)
                seen = np.zeros(len(rows), dtype=bool)
                for other in range(world):
                    order = list(range(other, len(rows), world))
                    for offset in range(0, len(order), args.chunk_size):
                        indices = order[offset:offset + args.chunk_size]
                        path = args.out / 'chunks' / name / f'rank{other}_offset{offset:07d}.npz'
                        if json.loads(path.with_suffix('.json').read_text())['sha256'] != digest(path):
                            raise ValueError('Prediction chunk failed final integrity check')
                        if seen[indices].any():
                            raise ValueError('Duplicate predictions')
                        probabilities[indices] = checked_predictions(path, indices, rows, classes)
                        seen[indices] = True
                if not seen.all():
                    raise ValueError('Missing predictions')
                labels = np.asarray([r['label'] for r in rows])
                subjects = np.asarray([r['subject'] for r in rows])
                metric = classification_metrics(labels, probabilities, classes)
                metric['per_subject'] = {s: classification_metrics(labels[subjects == s], probabilities[subjects == s], classes)
                                         for s in sorted(set(subjects))}
                metric['subject_macro_balanced_accuracy'] = float(np.mean([
                    m['balanced_accuracy'] for m in metric['per_subject'].values()]))
                save_predictions(args.out / f'train_{name}_predictions.npz', list(range(len(rows))), rows, probabilities)
                summaries[name] = metric
                write_json(args.out / 'metrics.json', dict(split='train', selected_step=identity['selected_step'],
                    complete=False, datasets=summaries, expected_counts=counts))
                report('dataset_complete', dataset=name, n=metric['n'], accuracy=metric['accuracy'],
                    balanced_accuracy=metric['balanced_accuracy'], completed_datasets=list(summaries))
            barrier()
        guard()
        unchanged()
        write_json(args.out / f'final_state_rank{rank}.json', dict(
            checkpoint_parameters_and_buffers_unchanged=True, test_evaluations=0, optimizer_steps=0))
        barrier()
        if rank == 0:
            result = dict(status='complete', split='train', selected_step=identity['selected_step'],
                checkpoint_sha256=identity['checkpoint_sha256'], datasets=summaries,
                total_samples=sum(m['n'] for m in summaries.values()),
                macro_accuracy=float(np.mean([m['accuracy'] for m in summaries.values()])),
                macro_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in summaries.values()])),
                test_evaluations=0, optimizer_steps=0, elapsed_seconds=time.monotonic() - started)
            if result['total_samples'] != identity['total_samples']:
                raise ValueError('Full training-set count mismatch')
            write_json(args.out / 'complete.json', result)
            report('complete', total_samples=result['total_samples'], macro_accuracy=result['macro_accuracy'])
    except BaseException as error:
        report('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, default=Path('artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/benchmark/leaf_jev_multitask_pilot_20261029_qc/train_eval'))
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=512)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out, args.training_run = args.out.resolve(), args.training_run.resolve()
    if args.microbatch < 1 or args.chunk_size < 1 or args.chunk_size % args.microbatch:
        raise ValueError('Positive chunk size must be divisible by microbatch')
    if args.out == args.training_run or args.training_run.is_relative_to(args.out):
        raise ValueError('Evaluation output must not replace the original training run')
    if args.worker:
        worker(args)
        return
    from eeg_align.multitask_data import write_json
    from eeg_align.runtime import CONFIG
    options = ['--training-run', str(args.training_run), '--out', str(args.out),
               '--microbatch', str(args.microbatch), '--chunk-size', str(args.chunk_size)]
    if args.start:
        args.out.mkdir(parents=True, exist_ok=args.resume)
        if (args.out / 'complete.json').exists():
            raise ValueError('Evaluation is already complete')
        with (args.out / 'controller.log').open('a') as log:
            process = subprocess.Popen([sys.executable, __file__, '--supervise'] + options,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.out / 'launch.json', dict(pid=process.pid, split='train', gpus=CONFIG['gpus'],
            original_checkpoint=str(args.training_run / 'best.pt'), command=options))
        print(json.dumps(dict(status='launched', output=str(args.out), pid=process.pid)))
    elif args.supervise:
        with (args.out / 'evaluation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, CONFIG['gpus'])), OMP_NUM_THREADS='3',
                OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node', str(len(CONFIG['gpus'])), __file__, '--worker'] + options
            with (args.out / 'evaluation.log').open('a') as log:
                result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write_json(args.out / 'controller_result.json', dict(exit_code=result.returncode,
                complete=(args.out / 'complete.json').exists(), test_evaluations=0))
            if result.returncode:
                raise SystemExit(result.returncode)
    else:
        parser.error('Use --start [--resume] to launch the isolated evaluation')


if __name__ == '__main__':
    main()
