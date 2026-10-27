"""Uncapped, read-only checkpoint evaluation in a version-matched worktree.

Run from the current repository; --runtime-root selects the historical imports.
The controller uses only the explicitly named GPUs and never trains or selects
checkpoints on test data. Existing output directories are deliberately rejected.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def prototype_texts(task):
    # Exact released LEAF class names where the task has a native counterpart.
    native = {
        'BNCI2014001': ['Left', 'Right', 'Foot', 'Tongue'],
        'BNCI2014004': ['Left', 'Right'],
        'BNCI2015001': ['Right', 'Foot'],
        'SEED': ['Negative', 'Neutral', 'Positive'],
        'EEGMAT': ['Resting', 'Workload'],
    }
    return native.get(task['task_id'], [c['option'] for c in task['classes']])


def shard_rows(rows, rank, world):
    """Stripe each dataset independently, covering every id exactly once."""
    groups = {}
    for row in rows:
        groups.setdefault(row['task'], []).append(row)
    return {tid: rr[rank::world] for tid, rr in groups.items()}


def audit(runtime):
    tasks = json.loads((runtime / 'configs/benchmark_tasks.json').read_text())['tasks']
    base = runtime / 'datasets/benchmark/signals/fold0'
    manifest = json.loads((base / 'complete.json').read_text())
    sets, subjects, tests = {}, {}, None
    for split in ('train', 'validation', 'test'):
        path = base / f'{split}.jsonl'
        assert sha256(path) == manifest['files_sha256'][split], split
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        sets[split] = {r['id'] for r in rows}
        subjects[split] = {(r['dataset'], r['subject']) for r in rows}
        assert len(sets[split]) == len(rows) == manifest['counts'][split]
        for row in rows:
            task = tasks[row['task']]
            assert row['dataset'] == task['task_id'] and row['split'] == split
            assert 0 <= row['label'] < len(task['classes'])
        if split == 'test':
            tests = rows
    for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
        assert not (sets[a] & sets[b]) and not (subjects[a] & subjects[b]), (a, b)
    counts = Counter(r['dataset'] for r in tests)
    assert set(counts) == {t['task_id'] for t in tasks}
    for task in tasks:
        ds = task['task_id']
        assert counts[ds] == manifest['datasets'][ds]['counts']['test']['n']
        source = runtime / 'datasets/benchmark/signals' / ds / 'complete.json'
        assert sha256(source) == manifest['datasets'][ds]['source_manifest_sha256']
    return tasks, tests, dict(test_n=len(tests), datasets=dict(counts), fold=0,
        manifest_sha256=sha256(base / 'complete.json'),
        all_catalog_datasets_present=True, sample_and_subject_splits_disjoint=True,
        omitted_benchmark_datasets={
            'SEED_VIG': 'Continuous regression; not supported by this choice checkpoint',
            'ThingsEEG2': 'Visual decoding/retrieval; no matching choice interface or trained checkpoint'},
        boundary='All test rows of the 11-task fold-0 manifest, NOT all LOSO/10-fold runs or all 13 benchmark datasets')


def encode_prototypes(tasks, runtime, upstream):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    texts = list(dict.fromkeys(s for t in tasks for s in prototype_texts(t)))
    old_text = (upstream / 'text_embeddings/mpnet-base.txt').read_text().splitlines()
    old_vectors = np.load(upstream / 'text_embeddings/mpnet-base.npy')
    cache = {s: old_vectors[old_text.index(s)] for s in texts if s in old_text}
    missing = [s for s in texts if s not in cache]
    if missing:
        folder = runtime / 'artifacts/benchmark/mpnet'
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = AutoModel.from_pretrained(folder, local_files_only=True).eval().requires_grad_(False)
        with torch.no_grad():
            x = tokenizer(missing, padding=True, return_tensors='pt')
            h = model(**x).last_hidden_state
            mask = x['attention_mask'][..., None]
            vectors = F.normalize((h * mask).sum(1) / mask.sum(1), dim=-1).numpy()
        cache.update(zip(missing, vectors))
        del model
    prototypes = [torch.from_numpy(np.stack([cache[s] for s in prototype_texts(t)])).cuda() for t in tasks]
    assert all(torch.allclose(p.norm(dim=-1), torch.ones(len(p), device='cuda'), atol=1e-5) for p in prototypes)
    return prototypes


def worker(args):
    os.chdir(args.runtime_root)
    sys.path.insert(0, str(args.runtime_root))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from eeg_align.leaf_runtime import CONFIG, UP
    from eeg_align.signal_data import SignalStore, load_rows
    from eeg_align.online_leaf import OnlineLEAF
    from eeg_align.checkpoint import restore_model, seed_rng, tensor_fingerprint
    from eeg_align.benchmark_data import metrics

    assert torch.cuda.device_count() == 1, 'Each worker must see exactly its assigned GPU'
    torch.cuda.set_device(0)
    torch.set_num_threads(3)
    seed_rng(42)
    plan = json.loads((args.out / 'plan.json').read_text())
    tasks, test, coverage = audit(args.runtime_root)
    assert coverage == plan['coverage']
    output = args.out / f'worker{args.worker}'
    output.mkdir(exist_ok=False)
    groups = shard_rows(test, args.worker, len(args.gpus))
    signals = SignalStore()
    for tid, arr in signals.arrays.items():
        assert tuple(arr.shape) == tuple(signals.meta[tid]['shape'])
    source_stat = args.checkpoint.stat()
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = state['config']
    assert state['step'] == plan['checkpoint_step']
    assert cfg['fold'] == 0 and cfg['bn_policy'] == 'frozen'
    assert cfg['data_manifest_sha256'] == coverage['manifest_sha256']
    assert cfg['task_sha256'] == sha256('configs/benchmark_tasks.json')
    assert cfg['instructions_sha256'] == sha256('artifacts/benchmark/instructions.npy')
    assert cfg['checkpoint'] == CONFIG['checkpoint'] and cfg['upstream_commit'] == CONFIG['upstream_commit']
    for path, expected in cfg['code_sha256'].items():
        assert sha256(path) == expected, f'Historical code drift: {path}'
    write_json(output / 'runtime.json', dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        torch=torch.__version__, transformers=transformers.__version__, checkpoint_step=state['step'],
        native_prototypes={t['task_id']: prototype_texts(t) for t in tasks}, code_hashes_verified=True))

    def progress(**values):
        write_json(output / 'progress.json', dict(time=time.time(), **values))

    def evaluate(name, predict):
        start = time.monotonic()
        completed = 0
        progress(model=name, status='starting', completed=0, total=sum(map(len, groups.values())))
        with (output / f'{name}.jsonl').open('x', buffering=1) as handle:
            for tid, rr in groups.items():
                count = len(tasks[tid]['classes'])
                for offset in range(0, len(rr), args.batch_size):
                    rows = rr[offset:offset + args.batch_size]
                    normal, reverse = predict(rows)
                    for row, p, q in zip(rows, normal, reverse):
                        p, q = np.asarray(p)[:count], np.asarray(q)[:count]
                        assert np.isfinite(p).all() and np.isfinite(q).all()
                        assert abs(p.sum()-1) < 1e-4 and abs(q.sum()-1) < 1e-4
                        handle.write(json.dumps(dict(row, probabilities=p.tolist(),
                            alternate_probabilities=q.tolist()), allow_nan=False) + '\n')
                    completed += len(rows)
                    if offset % (args.batch_size * 10) == 0 or offset + len(rows) == len(rr):
                        elapsed = time.monotonic() - start
                        progress(model=name, status='running', dataset=tasks[tid]['task_id'],
                            dataset_completed=offset+len(rows), dataset_total=len(rr), completed=completed,
                            total=sum(map(len, groups.values())), elapsed_s=elapsed,
                            samples_per_second=completed/max(elapsed, 1e-9))
                print(name, tasks[tid]['task_id'], len(rr), 'complete', flush=True)
        progress(model=name, status='complete', completed=completed, elapsed_s=time.monotonic()-start)

    if args.reuse_native:
        source = args.reuse_native / f'worker{args.worker}' / 'native.jsonl'
        assert sha256(source) == plan['reused_native_sha256'][str(args.worker)]
        previous = [json.loads(line) for line in source.read_text().splitlines()]
        wanted = [r for rr in groups.values() for r in rr]
        assert len(previous) == len(wanted)
        assert all(all(p[k] == v for k,v in r.items()) for p,r in zip(previous,wanted))
        (output / 'native.jsonl').symlink_to(source)
        write_json(output / 'native_reuse.json', dict(source=str(source), n=len(previous), sha256=sha256(source)))
    elif args.models in ('both', 'native'):
        progress(model='native', status='loading')
        encoder = OnlineLEAF('frozen').cuda().eval().requires_grad_(False)
        prototypes = encode_prototypes(tasks, args.runtime_root, UP)
        buffers_before = tensor_fingerprint(dict(encoder.named_buffers()))

        @torch.no_grad()
        def native_predict(rows):
            result = [None] * len(rows)
            for tid, indices, pieces in signals.batch(rows):
                embeddings = []
                inst = encoder.instructions[tid:tid+1].expand(len(indices), -1)
                for piece in pieces:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        emb, _ = encoder.leaf(torch.from_numpy(piece.copy()).cuda(), inst)
                    embeddings.append(emb.float())
                # Equal approximately-10s pieces; normalize the mean embedding.
                emb = F.normalize(torch.stack(embeddings).mean(0), dim=-1)
                probs = (emb @ prototypes[tid].T).softmax(-1).cpu().numpy()
                for index, p in zip(indices, probs):
                    result[index] = p
            # Similarity matching is order-equivariant by construction.
            return result, result

        evaluate('native', native_predict)
        assert tensor_fingerprint(dict(encoder.named_buffers())) == buffers_before
        del encoder, prototypes
        gc.collect()
        torch.cuda.empty_cache()

    if args.models in ('both', 'jev'):
        from eeg_align.jev import load_jev, JEVBridge
        from eeg_align.train_joint import make_variant
        progress(model='jev', status='loading')
        text, weight, bias, tokenizer = load_jev()
        model = JEVBridge(text, weight, bias, tokenizer, True, True, head_choices=16, input_dim=256)
        model.add_module('eeg_encoder', OnlineLEAF('frozen').cuda())
        restore_model(model, state)  # Strict name/shape/dtype check before freezing.
        # Preserve the historical requires_grad flags; @no_grad prevents graph
        # creation. Verify loaded tensors exactly, independently of forward numerics.
        assert tensor_fingerprint({n:p for n,p in model.named_parameters() if p.requires_grad}) == tensor_fingerprint(state['trainable'])
        assert tensor_fingerprint(dict(model.named_buffers())) == tensor_fingerprint(state['buffers'])
        model.eval()
        buffers_before = tensor_fingerprint(dict(model.named_buffers()))
        step = int(state['step'])
        prompts = {}
        for tid, task in enumerate(tasks):
            for variant in (0, 1):
                q = make_variant(task, 0, '', step, 42, variant)
                # Target labels may not affect input text.
                assert all(q['suffix'] == make_variant(task, c, '', step, 42, variant)['suffix']
                           for c in range(len(task['classes'])))
                prompts[tid, variant] = (q, torch.tensor(tokenizer.encode(q['suffix'], add_special_tokens=False), device='cuda'))

        @torch.no_grad()
        def jev_predict(rows):
            inputs, valid = model.eeg_encoder(signals.batch(rows), len(rows), unfrozen=True)
            results = []
            for variant in (0, 1):
                qs = [prompts[r['task'], variant] for r in rows]
                logits = model(inputs, valid, suffix_ids=[q[1] for q in qs],
                    num_choices=[q[0]['num_choices'] for q in qs])['logits']
                pp = logits.softmax(-1).cpu().numpy()
                canonical = []
                for row, p, (q, _) in zip(rows, pp, qs):
                    ids = [c['class_id'] for c in tasks[row['task']]['classes']]
                    canonical.append([float(p[q['option_class_ids'].index(i)]) for i in ids])
                results.append(canonical)
            return results

        # Replay old validation-only examples before permitting any JEV test run.
        check = json.loads((args.checkpoint.parent / f'reload_check_{step:04d}.json').read_text())
        val = {r['id']: r for r in load_rows('validation', 0)}
        rows = [val[i] for i in check['sample_ids']]
        def replay():
            predicted = []
            for offset in range(0, len(rows), 32):  # Original diagnostic batch size.
                p, _ = jev_predict(rows[offset:offset+32])
                predicted.extend(p)
            return predicted
        predicted = replay()
        # A local checkpoint round trip must remain bit-exact. The historical
        # reference is a separate BF16 execution, so report its difference too.
        with torch.no_grad():
            model.head_weight[0, 0].add_(1)
            restore_model(model, state)
        repeated = replay()
        local_exact = all(np.array_equal(p, q) for p,q in zip(predicted, repeated))
        differences = [float(np.max(np.abs(np.array(p)-np.array(q[:len(p)]))))
                       for p, q in zip(predicted, check['probabilities'])]
        agreement = float(np.mean([np.argmax(p) == np.argmax(q) for p, q in zip(predicted, check['probabilities'])]))
        write_json(output / 'checkpoint_replay.json', dict(samples=len(rows),
            max_abs_probability_difference=max(differences), prediction_agreement=agreement,
            tolerance=0.005, tensors_match_checkpoint_exactly=True, local_roundtrip_bit_exact=local_exact,
            reference_scope='cross-run BF16 probabilities, not a bit-exact historical execution claim',
            source='old validation-only checkpoint reload check'))
        assert local_exact and max(differences) < 0.005 and agreement == 1, 'Checkpoint replay failed; no JEV test evaluation permitted'
        del state
        print('CHECKPOINT_REPLAY_OK', len(rows), max(differences), flush=True)
        evaluate('jev', jev_predict)
        assert tensor_fingerprint(dict(model.named_buffers())) == buffers_before
    after = args.checkpoint.stat()
    assert (source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    for path, expected in cfg['code_sha256'].items():
        assert sha256(path) == expected
    assert sha256(__file__) == plan['evaluation_script_sha256']
    write_json(output / 'complete.json', dict(status='complete', pid=os.getpid(), checkpoint_unchanged=True,
        eval_buffers_unchanged=True, models=args.models, samples=sum(map(len, groups.values()))))


def merge(args, tasks, expected):
    import numpy as np
    os.chdir(args.runtime_root)
    sys.path.insert(0, str(args.runtime_root))
    from eeg_align.benchmark_data import metrics
    report = {}
    names = ['native', 'jev'] if args.models == 'both' else [args.models]
    for name in names:
        rows = []
        for rank in range(len(args.gpus)):
            folder = args.out / f'worker{rank}'
            assert (folder / 'complete.json').is_file()
            rows.extend(json.loads(line) for line in (folder / f'{name}.jsonl').read_text().splitlines())
        assert len(rows) == len(expected) == len({r['id'] for r in rows})
        by_id = {r['id']: r for r in rows}
        assert set(by_id) == {r['id'] for r in expected}
        rows = [by_id[r['id']] for r in expected]
        for actual, source in zip(rows, expected):
            assert all(actual[k] == v for k, v in source.items())
        width = max(len(t['classes']) for t in tasks)
        p = np.array([r['probabilities'] + [0.]*(width-len(r['probabilities'])) for r in rows])
        q = np.array([r['alternate_probabilities'] + [0.]*(width-len(r['alternate_probabilities'])) for r in rows])
        report[name] = dict(normal=metrics(rows, p), reversed=metrics(rows, q),
            consistency={t['task_id']: float(np.mean(p[[r['task']==tid for r in rows]].argmax(1) ==
                q[[r['task']==tid for r in rows]].argmax(1))) for tid,t in enumerate(tasks)})
        with (args.out / f'{name}_predictions.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
    write_json(args.out / 'results.json', report)
    lines = ['# Full fold-0 test evaluation', '',
        'All 60,275 retained test windows/trials in the 11-task manifest; no cap. Not all LOSO folds.', '',
        'Frozen checkpoint selection: prior validation-selected JEV step 600; no test selection.', '',
        'Native LEAF uses released weights and fixed prototypes with matched benchmark preprocessing; '
        'this is not a reproduction of a published native LEAF benchmark. Native softmax scores are not calibrated probabilities.', '',
        '| Dataset | N | Native ACC | Native BA | JEV ACC | JEV BA | JEV reversed ACC | Agreement |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task in tasks:
        ds = task['task_id']
        n = sum(r['dataset'] == ds for r in expected)
        def value(name, metric, order='normal'):
            return f"{100*report[name][order][ds]['epoch'][metric]:.2f}%" if name in report else '—'
        consistency = f"{100*report['jev']['consistency'][ds]:.2f}%" if 'jev' in report else '—'
        lines.append(f"| {ds} | {n} | {value('native','accuracy')} | {value('native','balanced_accuracy')} | "
            f"{value('jev','accuracy')} | {value('jev','balanced_accuracy')} | {value('jev','accuracy','reversed')} | {consistency} |")
    lines += ['', 'SEED-VIG regression and ThingsEEG2 retrieval are not supported by this checkpoint. '
        'Only fold 0 is covered. LEAF pretraining overlap is not excluded; this is not proof of unseen-dataset zero-shot transfer.', '']
    (args.out / 'RESULTS.md').write_text('\n'.join(lines))


def controller(args):
    import torch
    assert len(set(args.gpus)) == len(args.gpus)
    tasks, rows, coverage = audit(args.runtime_root)
    args.out.mkdir(parents=True, exist_ok=False)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    plan = dict(checkpoint=str(args.checkpoint), checkpoint_step=int(state['step']),
        checkpoint_sha256=sha256(args.checkpoint), selection='best.pt selected by prior capped validation macro BA, before this test run',
        runtime_root=str(args.runtime_root), runtime_commit=subprocess.check_output(
            ['git','-C',str(args.runtime_root),'rev-parse','HEAD'], text=True).strip(),
        evaluation_script_sha256=sha256(__file__), coverage=coverage, gpus=args.gpus, models=args.models,
        batch_size=args.batch_size, created_at=time.time(),
        native_protocol=dict(instruction='task-only, same as JEV training', prototypes={t['task_id']:prototype_texts(t) for t in tasks},
            long_epoch_pooling='normalize(mean(unit embeddings of approximately equal <=10s pieces))',
            scores='softmax(cosine), no fitted temperature; uncalibrated', preprocessing='same SignalStore as 11-task JEV training',
            pretraining_overlap='not excluded; no strict unseen-data zero-shot claim'),
        test_selection=False, train_steps=0)
    if args.reuse_native:
        previous = json.loads((args.reuse_native / 'plan.json').read_text())
        assert previous['coverage'] == coverage and previous['native_protocol'] == plan['native_protocol']
        assert previous['gpus'] == args.gpus
        plan['reused_native_source'] = str(args.reuse_native)
        plan['reused_native_sha256'] = {str(i): sha256(args.reuse_native / f'worker{i}' / 'native.jsonl') for i in range(len(args.gpus))}
    assert plan['checkpoint_step'] == 600, 'This run is predeclared for the prior validation-selected checkpoint'
    del state
    write_json(args.out / 'plan.json', plan)
    processes, handles = [], []
    try:
        for rank, gpu in enumerate(args.gpus):
            log = (args.out / f'worker{rank}.log').open('x')
            handles.append(log)
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--runtime-root', str(args.runtime_root),
                '--checkpoint', str(args.checkpoint), '--out', str(args.out), '--gpus', *map(str,args.gpus),
                '--models', args.models, '--batch-size', str(args.batch_size), '--worker', str(rank)]
            if args.reuse_native:
                cmd += ['--reuse-native', str(args.reuse_native)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='3',
                TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        def stop(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            codes = [p.poll() for p in processes]
            write_json(args.out / 'pipeline.json', dict(status='running', controller_pid=os.getpid(),
                workers=[dict(pid=p.pid, gpu=g, returncode=c) for p,g,c in zip(processes,args.gpus,codes)], updated=time.time()))
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError(f'Worker failed: {codes}; see worker logs')
            if all(c == 0 for c in codes):
                break
            time.sleep(15)
        merge(args, tasks, rows)
        assert sha256(args.checkpoint) == plan['checkpoint_sha256']
        write_json(args.out / 'pipeline.json', dict(status='complete', controller_pid=os.getpid(), finished=time.time(), coverage=coverage))
    except BaseException as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        write_json(args.out / 'pipeline.json', dict(status='failed', error=repr(error), finished=time.time()))
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[2,3])
    parser.add_argument('--models', choices=['native','jev','both'], default='both')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--reuse-native', type=Path,
        help='Reuse complete native predictions from a previous attempt with identical protocol and shards')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    if args.reuse_native:
        args.reuse_native = args.reuse_native.resolve()
        assert args.models == 'both'
    assert args.batch_size > 0
    if args.preflight:
        print(json.dumps(audit(args.runtime_root)[2], indent=2))
    elif args.worker is None:
        controller(args)
    else:
        assert 0 <= args.worker < len(args.gpus)
        worker(args)


if __name__ == '__main__':
    main()
