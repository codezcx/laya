"""LEAF-only alignment (two GPUs in parallel), then matched four-GPU JEV SFT."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.contrastive_protocol import CONFIG, DATA, code_identity, load_text_bank
from eeg_align.multitask_data import write_json
from eeg_align.runtime import digest
from scripts.run_benchmark import require_gpu_capacity, training_command, verify_initialization


def verify_baseline_recipe():
    """Do not silently change the downstream optimizer, sampling or input recipe."""
    baseline = Path(CONFIG['baseline_root'])
    completion = json.loads((baseline / 'complete.json').read_text())
    if completion['test_evaluations'] or completion['updates'] != DATA['steps']:
        raise ValueError('No compatible completed no-alignment baseline')
    old = json.loads((baseline / 'config.json').read_text())
    exempt = {'arms', 'arm_settings'}
    if {k:v for k,v in old['config'].items() if k not in exempt} != {k:v for k,v in DATA.items() if k not in exempt}:
        raise ValueError('Downstream recipe changed beyond the aligned initialization')
    for path in ('eeg_align/leaf_data.py', 'eeg_align/online_leaf.py', 'eeg_align/jev.py',
                 'eeg_align/multitask_data.py', 'eeg_align/evaluation.py', 'eeg_align/lr_schedule.py',
                 'configs/benchmark_tasks.json'):
        if digest(path) != old['code_sha256'][path]:
            raise ValueError(f'Matched input/optimizer/model semantics changed: {path}')
    return json.loads((baseline / 'initial_state.json').read_text())


def verify_alignment_initializations(left, right):
    for key in ('encoder_sha256', 'projection_sha256', 'text_sha256', 'buffer_sha256'):
        if left[key] != right[key]:
            raise ValueError(f'Alignment initialization mismatch: {key}')


def validate_finished(directory, phase, smoke):
    identity = json.loads((directory / 'config.json').read_text())
    if identity['smoke'] != smoke or identity['config'] != (CONFIG if phase == 'alignment' else DATA):
        raise ValueError('Existing run belongs to another configuration')
    if any(not Path(p).is_file() or digest(p) != h for p,h in identity['code_sha256'].items()):
        raise ValueError('Existing run source identity changed')
    done = json.loads((directory / 'complete.json').read_text())
    if done['test_evaluations'] or not done['checkpoint_reload_verified']:
        raise ValueError('Incomplete serialization/test-use contracts')
    if phase == 'alignment':
        if not done['encoder_updated'] or not done['frozen_text_verified'] or done['jev_loaded']:
            raise ValueError('Alignment contract failed')
    elif (not done['encoder_update_contract_verified'] or not done['rank_weights_verified']
          or not done['distributed_sampling_verified'] or done['world_size'] != DATA['world_size']):
        raise ValueError('SFT synchronization/encoder contract failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--start', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    baseline = verify_baseline_recipe()
    load_text_bank()
    output = Path('artifacts/benchmark') / CONFIG['run_name']
    if args.start:
        if args.resume:
            if not output.is_dir():
                raise ValueError('No pipeline to resume')
            with (output / 'controller.lock').open('a') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if json.loads((output / 'pipeline.json').read_text())['status'] == 'complete':
                raise ValueError('Pipeline is already complete')
        else:
            output.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', __file__, '--execute'] + (['--resume'] if args.resume else [])
        with (output / ('controller.log' if not args.resume else f'controller_resume_{time.time_ns()}.log')).open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
        print(json.dumps(dict(controller_pid=process.pid, output=str(output))))
        return
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = code_identity()
    state = dict(controller_pid=os.getpid(), test_evaluations=0, run_name=CONFIG['run_name'])
    def report(**values):
        state.update(values, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        write_json(output / 'pipeline.json', state)
        print(json.dumps(state), flush=True)
    def guard():
        if code_identity() != hashes:
            raise RuntimeError('Live pipeline code/config changed')
    def launch(directory, command, gpus, phase, smoke):
        guard()
        if (directory / 'complete.json').exists():
            if not args.resume:
                raise FileExistsError(directory)
            validate_finished(directory, phase, smoke)
            return None
        if directory.exists():
            if not args.resume or not (directory / 'last.pt').exists():
                raise ValueError(f'Uncheckpointed/unauthorized existing run: {directory}')
            command.append('--resume')
        directory.parent.mkdir(parents=True, exist_ok=True)
        require_gpu_capacity(gpus, 12000 if phase == 'alignment' else DATA['minimum_free_gpu_mib'])
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
        filename = f'{phase}_{"smoke" if smoke else "train"}_{directory.name}'
        if args.resume:
            filename += f'_resume_{time.time_ns()}'
        with (output / f'{filename}.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=env, start_new_session=True)
        return process
    def await_jobs(jobs, phase, smoke):
        report(status=f'{phase}_{"smoke" if smoke else "train"}',
               processes={p.name:(child.pid if child else None) for p,child in jobs})
        while any(child is not None and child.poll() is None for _,child in jobs):
            time.sleep(10)
        for directory, child in jobs:
            if child is not None and child.returncode:
                raise RuntimeError(f'{directory}: exited {child.returncode}; see phase log')
            validate_finished(directory, phase, smoke)
        guard()
    try:
        report(status='preflight', code_sha256=hashes,
               launch_free_mib=require_gpu_capacity(CONFIG['gpus'], DATA['minimum_free_gpu_mib']))
        preflight = output / 'signal_preflight.json'
        if not preflight.exists():
            subprocess.run([sys.executable, '-u', 'scripts/check_multitask_signals.py',
                '--out', str(preflight), '--global-batch', str(CONFIG['global_batch']),
                '--microbatch', str(DATA['microbatch'])], check=True)
        proof = json.loads(preflight.read_text())
        if (not proof['passed'] or proof['global_batch'] != CONFIG['global_batch']
                or proof['planned_updates'] != CONFIG['steps']
                or any(digest(p) != h for p,h in proof['code_sha256'].items())):
            raise ValueError('CPU preflight does not match current planned inputs')
        # sample(batch=16) is exactly the first 16 rows of sample(batch=32),
        # and microbatch=4 is the actual downstream rank partition (unit-tested).
        for smoke in (True, False):
            jobs = []
            for arm, gpu in zip(CONFIG['arms'], CONFIG['alignment_gpus'], strict=True):
                directory = output / 'alignment' / ('smoke' if smoke else 'train') / arm
                command = [sys.executable, '-u', '-m', 'eeg_align.train_contrastive',
                           '--arm', arm, '--out', str(directory), '--bank-root', CONFIG['text_bank_root']]
                if smoke:
                    command.append('--smoke')
                jobs.append((directory, launch(directory, command, [gpu], 'alignment', smoke)))
            await_jobs(jobs, 'alignment', smoke)
            inits = [json.loads((p / 'initial_state.json').read_text()) for p,_ in jobs]
            verify_alignment_initializations(*inits)
            if not smoke:
                for directory,_ in jobs:
                    old = output / 'alignment' / 'smoke' / directory.name / 'initial_state.json'
                    verify_alignment_initializations(json.loads(old.read_text()), json.loads((directory/'initial_state.json').read_text()))
        for objective, arm in CONFIG['downstream_arms'].items():
            for smoke in (True, False):
                directory = output / ('smoke' if smoke else 'train') / arm
                command = training_command(sys.executable, CONFIG['gpus'], arm, directory, smoke)
                child = launch(directory, command, CONFIG['gpus'], 'jev', smoke)
                await_jobs([(directory,child)], 'jev', smoke)
                init = json.loads((directory/'initial_state.json').read_text())
                for key in ('common_connector_sha256', 'jev_adapter_head_sha256', 'world_size'):
                    if init[key] != baseline[key]:
                        raise ValueError(f'Downstream initialization differs from baseline: {key}')
                if not init['aligned_transfer'] or init['aligned_transfer']['objective'] != objective:
                    raise ValueError('Missing aligned Tower provenance')
            verify_initialization(json.loads((output/'smoke'/arm/'initial_state.json').read_text()), init)
            if not (output/'figures'/arm/'training_validation_curves.png').exists():
                subprocess.run([sys.executable, 'scripts/plot_training_curves.py',
                                '--run-dir', str(output), '--arm', arm], check=True)
        report(status='complete', baseline_initialization_verified=True, smoke_weights_not_reused=True)
    except BaseException as exc:
        report(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
