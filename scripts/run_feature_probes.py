"""Detached CPU-only probe workers, immutable provenance, live result summary."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from eeg_probes.data import DATASETS, digest, save_json, verify_source_cache

SOURCES = ['eeg_probes/__init__.py', 'eeg_probes/data.py', 'eeg_probes/train.py',
           'scripts/run_feature_probes.py', 'eeg_align/cache.py', 'eeg_tasks/data.py',
           'eeg_tasks/normalization.py', 'eeg_tasks/channels.py', 'configs/data.json',
           'configs/encoder.json', 'docs/FEATURE_PROBES.md']


def orchestrate(root):
    from eeg_probes.train import summarize
    jobs = []
    try:
        save_json(root / 'status.json', dict(phase='verifying_source_cache'))
        audit = verify_source_cache()
        save_json(root / 'cache_audit.json', audit)
        for dataset in DATASETS:
            log = (root / f'{dataset}.log').open('x')
            env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
            process = subprocess.Popen([sys.executable, '-u', '-m', 'eeg_probes.train', '--dataset', dataset,
                                        '--root', str(root), '--device', 'cpu'],
                                       env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            jobs.append((dataset, process, log))
        save_json(root / 'status.json', dict(phase='training', workers={name: p.pid for name, p, _ in jobs},
                                            device='cpu', threads_per_worker=4))
        while any(process.poll() is None for _, process, _ in jobs):
            summarize(root)
            time.sleep(10)
        codes = {name: p.returncode for name, p, _ in jobs}
        summarize(root)
        save_json(root / 'status.json', dict(phase='complete' if all(c == 0 for c in codes.values()) else 'failed',
                                            returncodes=codes))
        if any(c != 0 for c in codes.values()):
            raise RuntimeError(codes)
    except BaseException as exc:
        save_json(root / 'status.json', dict(phase='failed', error=repr(exc)))
        traceback.print_exc()
        raise
    finally:
        for _, _, log in jobs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('artifacts/feature_probes'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.out
    if args.worker:
        orchestrate(root)
        return
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / 'source'
    snapshot.mkdir()
    for source in SOURCES:
        destination = snapshot / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    save_json(root / 'source_snapshot.json', dict(
        hashes={source: digest(source) for source in SOURCES},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
        note='Snapshot includes all runtime source hashes; unrelated dirty JEV files are not edited.',
        python=sys.version))
    save_json(root / 'status.json', dict(phase='launching'))
    with (root / 'orchestrator.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', __file__, '--out', str(root), '--worker'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    save_json(root / 'launcher.json', dict(pid=process.pid, output=str(root), device='cpu'))
    print(json.dumps(dict(launched=True, pid=process.pid, output=str(root), device='cpu')))


if __name__ == '__main__':
    main()
