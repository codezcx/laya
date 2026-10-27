"""Queue independent official Cross SFT folds on explicitly assigned GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from eeg_align.leaf_runtime import CONFIG, digest
from eeg_align.official_data import build_plan, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--gpus', nargs='+', type=int, default=CONFIG['gpus'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= set(CONFIG['gpus']):
        parser.error('Use distinct configured GPU indices only')
    plan = build_plan(CONFIG)
    chosen = [j for j in plan['jobs'] if
        (args.datasets is None or j['dataset'] in args.datasets) and
        (args.seeds is None or j['seed'] in args.seeds) and
        (args.folds is None or j['fold'] in args.folds)]
    if not chosen:
        parser.error('No prepared folds match selection')
    if args.datasets and set(args.datasets) - {j['dataset'] for j in chosen}:
        parser.error('Some selected datasets lack prepared splits')
    args.out.mkdir(parents=True, exist_ok=False)
    plan.update(selected_jobs=chosen, selected_runs=len(chosen), smoke=args.smoke,
                gpu_assignment=args.gpus, config=CONFIG)
    write_json(args.out / 'plan.json', plan)
    print('PLAN', json.dumps({k: plan[k] for k in ('planned_runs', 'planned_updates', 'selected_runs', 'pending')}), flush=True)
    if args.plan_only:
        return
    watched = list(Path('eeg_align').glob('*.py')) + [
        Path(__file__), Path('configs/benchmark_train.json'), Path('configs/benchmark_tasks.json'),
        Path('scripts/prepare_official_splits.py')]
    hashes = {str(p): digest(p) for p in watched}
    state = dict(status='running', controller_pid=os.getpid(), smoke=args.smoke,
                 completed=[], active={}, queued=chosen.copy(), selected_runs=len(chosen),
                 planned_runs=plan['planned_runs'], code_sha256=hashes)
    jobs = {}
    release_proofs = {}
    stopping = False

    def stop(signum=None, frame=None):
        nonlocal stopping
        stopping = True
        for proc, _, _ in jobs.values():
            if proc.poll() is None:
                proc.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while state['queued'] or jobs:
            if hashes != {str(p): digest(p) for p in watched}:
                raise RuntimeError('Live source guard changed')
            for _, _, info in jobs.values():
                proof_path = Path(info['out']) / 'initial_state.json'
                if proof_path.exists():
                    proof = json.loads(proof_path.read_text())
                    key = info['seed']
                    core = {k: proof[k] for k in ('trainable_sha256', 'buffers_sha256', 'encoder_sha256')}
                    if key in release_proofs and core != release_proofs[key]:
                        raise RuntimeError('Same-seed folds did not start from identical fresh weights')
                    release_proofs[key] = core
            state['verified_release_initialization_by_seed'] = release_proofs
            for gpu, (proc, handle, info) in list(jobs.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                handle.close()
                del jobs[gpu]
                del state['active'][str(gpu)]
                done = Path(info['out']) / 'complete.json'
                if not stopping and (rc != 0 or not done.exists()):
                    raise RuntimeError(f"Worker failed: {info['dataset']}, seed {info['seed']}, fold {info['fold']}; see {info['out']}.log")
                if done.exists():
                    state['completed'].append(dict(**info, result=json.loads(done.read_text())))
            if stopping:
                if not jobs:
                    break
            else:
                for gpu in args.gpus:
                    if gpu in jobs or not state['queued']:
                        continue
                    job = state['queued'].pop(0)
                    name = f"{job['dataset']}_seed{job['seed']}_fold{job['fold']:03d}"
                    folder = args.out / name
                    handle = (args.out / f'{name}.log').open('x')
                    cmd = [sys.executable, '-u', '-m', 'eeg_align.train_joint',
                           '--dataset', job['dataset'], '--seed', str(job['seed']),
                           '--fold', str(job['fold']), '--out', str(folder)]
                    if args.smoke:
                        cmd.append('--smoke')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='3',
                               OPENBLAS_NUM_THREADS='3', TOKENIZERS_PARALLELISM='false')
                    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
                    info = dict(**job, out=str(folder), pid=proc.pid, gpu=gpu)
                    jobs[gpu] = (proc, handle, info)
                    state['active'][str(gpu)] = info
            write_json(args.out / 'pipeline.json', state)
            if jobs:
                time.sleep(5)
        state['status'] = 'stopped' if stopping else 'complete'
        write_json(args.out / 'pipeline.json', state)
    except BaseException as error:
        stop()
        state.update(status='failed', error=repr(error))
        write_json(args.out / 'pipeline.json', state)
        raise
    finally:
        for proc, handle, _ in jobs.values():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            handle.close()


if __name__ == '__main__':
    main()
