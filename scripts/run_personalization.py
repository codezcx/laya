"""Detached four-worker, paired B/C personal calibration; no global retraining."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eeg_align.multitask_data import TrialStore, write_json
from eeg_align.personalization import CONFIG, ROOT, make_manifest, source_identity


def require_capacity():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    free = {int(i): int(m) for i,m in (row.split(',') for row in output.strip().splitlines())}
    if any(free.get(i, 0) < 60000 for i in CONFIG['gpus']):
        raise RuntimeError(f'Need 60000 MiB free on each of the four GPUs; found {free}')
    return free


def verify_paired(root, smoke=False):
    manifest = json.loads((root/('smoke_manifest.json' if smoke else 'manifest.json')).read_text())
    pairs = {}
    for case in manifest['cases']:
        path = root/('smoke' if smoke else 'episodes')/case['key']/'complete.json'
        result = json.loads(path.read_text())
        if result['smoke'] != smoke or result['case'] != case or result['code_sha256'] != source_identity():
            raise ValueError('Mismatched episode identity')
        if not all(result[k] for k in ('parent_encoder_frozen_verified', 'text_readout_frozen_verified',
                                       'jev_base_connector_head_frozen_verified', 'checkpoint_reload_verified',
                                       'reference_batch_ids_verified', 'reference_baseline_verified')):
            raise ValueError('Freeze/reload checks failed')
        if result['lm_lora_updated'] != case['train_lm_lora'] or result['objective'] != 'jev_ce':
            raise ValueError('Incorrect language update/objective')
        if result['query_used_for_selection'] or result['steps'] != (3 if smoke else CONFIG['steps']):
            raise ValueError('Invalid query use or training budget')
        key = (case['dataset'], case['subject'], case['shots_per_class'], case['seed'])
        pairs.setdefault(key, {})[case['arm']] = result
    for pair in pairs.values():
        if set(pair) != set(CONFIG['arms']):
            raise ValueError('Unpaired methods')
        first = pair[CONFIG['arms'][0]]
        for result in pair.values():
            for field in ('support_sha256', 'query_sha256'):
                if result['case'][field] != first['case'][field]:
                    raise ValueError('Arms used different trials')
            if (result['training_batch_ids_sha256'] != first['training_batch_ids_sha256']
                    or result['baseline'] != first['baseline']):
                raise ValueError('Arm batch stream or common baseline differs')
    return dict(episodes=len(manifest['cases']), paired_comparisons=len(pairs),
                frozen_and_reload_verified=True, same_support_query_batches_and_baseline=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(REPO)
    root = ROOT
    if (root/'cancelled.json').exists():
        raise RuntimeError('Personalization run was cancelled; no automatic restart')
    if (root/'complete.json').exists():
        raise RuntimeError('Run already complete; no automatic retraining')
    if args.start:
        if args.resume:
            if not root.is_dir():
                raise ValueError('No run to resume')
            lock = (root/'controller.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.close()
        else:
            root.mkdir(parents=True, exist_ok=False)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        log_path = root/f'controller_{stamp}.log'
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', __file__, '--execute']
        if args.resume:
            command.append('--resume')
        with log_path.open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid, root=str(root), log=str(log_path))))
        return
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = source_identity()
    state = dict(controller_pid=os.getpid(), gpus=CONFIG['gpus'], config=CONFIG, code_sha256=source)
    children = []
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def guard():
        if stopping or (root/'cancelled.json').exists():
            raise RuntimeError('Controller stopped; no further jobs will launch')
        if source != source_identity():
            raise RuntimeError('Source identity changed during execution')
    def report(**values):
        state.update(values, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        write_json(root/'pipeline.json', state)
        print(json.dumps(state), flush=True)
    def launch(mode, worker=None, smoke=False):
        guard()
        command = [sys.executable, '-u', '-m', 'eeg_align.run_personalization', mode]
        label = mode
        gpu = CONFIG['gpus'][0]
        if worker is not None:
            command += ['--worker', str(worker)]
            label += f'_{worker}'
            gpu = CONFIG['gpus'][worker]
        if smoke:
            command.append('--smoke')
            label = 'smoke_' + label
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log_path = root/f'{label}.log'
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return dict(pid=child.pid, gpu=gpu, log=str(log_path), mode=mode, worker=worker, smoke=smoke)
    def wait_children():
        while any(p.poll() is None for p in children):
            guard()
            failed = [p for p in children if p.poll() not in (None,0)]
            if failed:
                raise RuntimeError(f'Worker failed: {[(p.pid,p.returncode) for p in failed]}')
            time.sleep(2)
        failed = [p for p in children if p.returncode != 0]
        if failed:
            raise RuntimeError(f'Worker failed: {[(p.pid,p.returncode) for p in failed]}')
        children.clear()
    try:
        report(status='preflight')
        free = require_capacity()
        if shutil.disk_usage(root).free < CONFIG['minimum_free_disk_gib']*1024**3:
            raise RuntimeError('Insufficient free space for all personal LM adapters and reserve')
        manifest = make_manifest(TrialStore())
        report(status='preparing', episodes=len(manifest['cases']), subjects=len(manifest['cohorts']), free_mib=free)
        report(jobs=[launch('prepare')])
        wait_children()
        report(status='smoke', jobs=[launch('worker', i, True) for i in range(4)])
        wait_children()
        smoke_proof = verify_paired(root, True)
        write_json(root/'smoke_verified.json', smoke_proof)
        guard()
        require_capacity()
        report(status='formal', smoke_verified=smoke_proof, jobs=[launch('worker', i) for i in range(4)])
        wait_children()
        formal_proof = verify_paired(root)
        report(status='summarizing', paired_verified=formal_proof, jobs=[launch('summary')])
        wait_children()
        report(status='complete', jobs=[])
        write_json(root/'complete.json', state)
    except BaseException as error:
        # Only processes launched by this controller, never unrelated GPU work.
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic()+30
        while any(p.poll() is None for p in children) and time.monotonic()<deadline:
            time.sleep(1)
        for child in children:
            if child.poll() is None:
                child.kill()
        for child in children:
            child.wait()
        report(status='failed_or_stopped', error=repr(error), jobs=[])
        raise


if __name__ == '__main__':
    main()
