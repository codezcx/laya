"""Detached matched pilot: CPU audit -> all-arm smoke -> queued GPU training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.multitask_data import write_json
from eeg_align.encoder_lora import verify_matched_recipe, verify_control_initialization


def require_gpu_capacity(gpus, minimum_free_mib):
    """Permit user-authorized co-tenancy; reserve headroom, never kill peers."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    free = {int(i): int(m) for i, m in (r.split(',') for r in rows.strip().splitlines())}
    if any(free.get(g, 0) < minimum_free_mib for g in gpus):
        raise RuntimeError(f'Insufficient GPU headroom: need {minimum_free_mib} MiB free; '
                           f'observed={ {g: free.get(g) for g in gpus} }')
    return {g: free[g] for g in gpus}


def verify_initializations(proofs, config):
    control = verify_matched_recipe(config)
    for key in ('common_connector_sha256', 'jev_adapter_head_sha256'):
        if len({p[key] for p in proofs.values()}) != 1:
            raise RuntimeError(f'Arm initialization mismatch: {key}')
    for arm, proof in proofs.items():
        verify_control_initialization(proof, control)
        setting = config['arm_settings'][arm]
        if proof['initial_encoder_sha256'] != config['aligned_sources'][setting['aligned_source']]['encoder_sha256']:
            raise RuntimeError(f'Wrong aligned encoder initialization: {arm}')
    for readout in {s['readout'] for s in config['arm_settings'].values()}:
        hashes = {proofs[a]['readout_sha256'] for a in config['arms']
                  if config['arm_settings'][a]['readout'] == readout}
        if len(hashes) != 1:
            raise RuntimeError(f'Unmatched {readout} initialization')


def run_on_gpu_slots(arms, gpus, launch, report, *, poll_seconds=15, running=None):
    """Run every arm exactly once, at most one process per assigned GPU.

    On failure, drain live independent arms but never launch additional arms.
    ``running`` stays visible to the caller for failure reporting, not killing.
    """
    if not arms or not gpus or len(set(arms)) != len(arms) or len(set(gpus)) != len(gpus):
        raise ValueError('Arms and GPU slots must be nonempty and unique')
    pending, available = list(arms), list(gpus)
    running = [] if running is None else running
    if running:
        raise ValueError('New phase cannot inherit live children')
    processes, failures = {}, {}
    while pending or running:
        for item in list(running):
            arm, gpu, process = item
            code = process.poll()
            if code is not None:
                running.remove(item)
                available.append(gpu)
                processes[arm]['returncode'] = code
                if code:
                    failures[arm] = code
        while pending and available and not failures:
            arm, gpu = pending.pop(0), available.pop(0)
            process = launch(arm, gpu)
            running.append((arm, gpu, process))
            processes[arm] = dict(pid=process.pid, gpu=gpu, returncode=None)
        report(processes=processes, queued_arms=pending.copy(), failures=failures.copy())
        if failures and not running:
            raise RuntimeError(f'GPU job failures: {failures}; unstarted arms: {pending}')
        if running:
            time.sleep(poll_seconds)
    return processes


def unfinished_arms(output, config):
    """Resume only this recipe, without changing checkpoint identity for new GPUs."""
    pending = []
    for arm in config['arms']:
        directory = output / 'train' / arm
        identity = json.loads((directory / 'config.json').read_text())
        if identity['arm'] != arm or identity['config'] != config or identity['smoke']:
            raise ValueError(f'Resume identity mismatch: {arm}')
        if any(not Path(p).is_file() or digest(p) != h
               for p, h in identity['code_sha256'].items()):
            raise ValueError(f'Resume code hash mismatch: {arm}')
        if (directory / 'complete.json').is_file():
            continue
        if not all((directory / name).is_file() for name in ('last.pt', 'best.pt')):
            raise ValueError(f'Resume requires last.pt and best.pt: {arm}')
        pending.append(arm)
    return pending


def snapshot_interrupted_outputs(output, destination, arms):
    """Keep interrupted audits/predictions; checkpoint files are never moved."""
    if (output / 'pipeline.json').is_file():
        shutil.copy2(output / 'pipeline.json', destination / 'previous_pipeline.json')
    for arm in arms:
        source = output / 'train' / arm
        target = destination / 'before_resume' / arm
        target.mkdir(parents=True, exist_ok=False)
        for path in source.iterdir():
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.npz'}:
                shutil.copy2(path, target / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Recover unfinished formal arms; skip smoke')
    parser.add_argument('--gpus', type=int, nargs='+', help='GPU slots for this launch only; saved recipe is unchanged')
    parser.add_argument('--recovery-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path('artifacts/benchmark') / CONFIG['run_name']
    gpus = CONFIG['gpus'] if args.gpus is None else args.gpus
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        parser.error('GPU slots must be nonempty, unique, and nonnegative')
    if args.start:
        if args.resume:
            unfinished_arms(output, CONFIG)
            # Check the shared controller lock before creating recovery artifacts.
            with (output / 'controller.lock').open('a') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            args.recovery_dir = output / 'recovery' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'_{os.getpid()}')
            args.recovery_dir.mkdir(parents=True, exist_ok=False)
            log_path = args.recovery_dir / 'controller.log'
        else:
            output.mkdir(parents=True, exist_ok=False)
            log_path = output / 'controller.log'
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [sys.executable, '-u', __file__, '--execute', '--gpus', *map(str, gpus)]
        if args.resume:
            command += ['--resume', '--recovery-dir', str(args.recovery_dir)]
        with log_path.open('x') as log:
            process = subprocess.Popen(command,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env, cwd=ROOT)
        print(json.dumps(dict(controller_pid=process.pid, output=str(output), log=str(log_path), gpus=gpus)))
        return
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='preflight', test_evaluations=0, controller_pid=os.getpid())
    def report(**values):
        state.update(values, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        write_json(output / 'pipeline.json', state)
        print(json.dumps(state), flush=True)
    running = []
    try:
        pending = CONFIG['arms']
        if args.resume:
            if args.recovery_dir is None or not args.recovery_dir.is_dir():
                raise ValueError('Start recovery with --start --resume')
            pending = unfinished_arms(output, CONFIG)
            snapshot_interrupted_outputs(output, args.recovery_dir, pending)
            state.update(resumed=True, recovery_directory=str(args.recovery_dir),
                         previously_complete=[a for a in CONFIG['arms'] if a not in pending])
        watched = sorted((ROOT / 'eeg_align').glob('*.py')) + [ROOT / p for p in (
            'configs/benchmark_train.json', 'configs/benchmark_tasks.json',
            'scripts/run_benchmark.py', 'scripts/check_multitask_signals.py',
            'scripts/prepare_multitask_splits.py', 'scripts/prepare_multitask_signals.py',
            'scripts/audit_multitask_signals.py')]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in watched}
        def guard():
            if (hashes != {str(p.relative_to(ROOT)): digest(p) for p in watched}
                    or json.loads((ROOT / 'configs/benchmark_train.json').read_text()) != CONFIG):
                raise RuntimeError('Queued experiment code/config changed; refusing new launches')
        guard()
        control = verify_matched_recipe(CONFIG)
        if control is not None:
            report(matched_control=control['directory'], matched_recipe_verified=True)
        report(code_sha256=hashes, configured_arms=CONFIG['arms'], assigned_gpu_slots=gpus)
        free = require_gpu_capacity(gpus, CONFIG.get('minimum_free_gpu_mib', 75000))
        report(launch_free_mib=free, gpu_policy='headroom check; existing peer jobs are never stopped')
        if not args.resume:
            report(status='cpu_signal_preflight')
            subprocess.run([sys.executable, '-u', 'scripts/check_multitask_signals.py',
                '--out', str(output / 'signal_preflight.json')], check=True)
        initializations = ({arm: json.loads((output / 'smoke' / arm / 'initial_state.json').read_text())
                            for arm in CONFIG['arms']} if args.resume else {})
        for phase in (('train',) if args.resume else ('smoke','train')):
            running = []
            def launch(arm, gpu):
                guard()
                require_gpu_capacity([gpu], CONFIG.get('minimum_free_gpu_mib', 75000))
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                directory = output / phase / arm
                directory.parent.mkdir(exist_ok=True)
                command = [sys.executable, '-u', '-m', 'eeg_align.train_joint', '--arm', arm, '--out', str(directory)]
                if phase == 'smoke': command.append('--smoke')
                if args.resume: command.append('--resume')
                log_root = args.recovery_dir if args.resume else output
                with (log_root / f'{phase}_{arm}.log').open('x') as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, env=env, start_new_session=True)
                return child
            phase_arms = pending if args.resume else CONFIG['arms']
            if phase_arms:
                run_on_gpu_slots(phase_arms, gpus, launch,
                    lambda **values: report(status=phase + ('_partial_failure' if values['failures'] else ''),
                                            **values), running=running)
            guard()
            proofs = {arm: json.loads((output / phase / arm / 'initial_state.json').read_text()) for arm in CONFIG['arms']}
            verify_initializations(proofs, CONFIG)
            if phase == 'smoke':
                initializations = proofs
            else:
                for arm, proof in proofs.items():
                    for key in ('common_connector_sha256', 'jev_adapter_head_sha256', 'initial_encoder_sha256', 'readout_sha256'):
                        if proof[key] != initializations[arm][key]:
                            raise RuntimeError(f'Smoke weights leaked into formal training: {arm}/{key}')
            sequences = []
            for arm in CONFIG['arms']:
                complete = json.loads((output / phase / arm / 'complete.json').read_text())
                if (complete['test_evaluations'] or not complete['checkpoint_reload_verified']
                        or not complete['encoder_update_contract_verified']):
                    raise RuntimeError('Unsafe/incomplete preflight')
                records = [json.loads(line) for line in (output / phase / arm / 'metrics.jsonl').read_text().splitlines()]
                sequences.append([(r['step'], r['batch_ids_sha256']) for r in records if r['status'] == 'training'])
            if any(s != sequences[0] for s in sequences[1:]):
                raise RuntimeError('Arms saw different training samples')
            report(status=phase+'_complete', matched_jev_initialization=True,
                   matched_connector_initialization=True, matched_readout_initialization=True,
                   sampling_matched=True, aligned_encoder_initialization_verified=True)
        for arm in CONFIG['arms']:
            subprocess.run([sys.executable, 'scripts/plot_training_curves.py',
                            '--run-dir', str(output), '--arm', arm], check=True)
        report(status='complete', selection='Compare full validation before separately authorizing test')
    except BaseException as exc:
        report(status='failed', error=f'{type(exc).__name__}: {exc}',
               remaining_processes={arm:p.pid for arm,_,p in running if p.poll() is None})
        raise


if __name__ == '__main__':
    main()
