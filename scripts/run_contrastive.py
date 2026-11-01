"""Detached matched VideoMAE/EEG-to-text objective controls on GPU 6 and 7."""
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
from eeg_contrastive.protocol import CONFIG, code_identity
from eeg_align.multitask_data import write_json
from scripts.run_benchmark import run_on_gpu_slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path('artifacts/benchmark')/CONFIG['run_name']
    if args.start:
        output.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (output/'controller.log').open('x') as log:
            child = subprocess.Popen([sys.executable,'-u',__file__,'--execute'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid, output=str(output))))
        return
    lock = (output/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = dict(status='preflight', controller_pid=os.getpid(), test_evaluations=0, jev_loaded=False)
    def report(**values):
        state.update(values, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
        write_json(output/'pipeline.json',state)
        print(json.dumps(state),flush=True)
    hashes = code_identity()
    def guard():
        if code_identity() != hashes:
            raise RuntimeError('Alignment source/config changed; not launching a new phase')
    def require_free(gpus):
        records = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used = {int(i):int(m) for i,m in (line.split(',') for line in records.strip().splitlines())}
        if any(used.get(g,999999)>1000 for g in gpus):
            raise RuntimeError(f'Assigned GPUs are no longer empty: {gpus}')
    running = []
    try:
        require_free(CONFIG['gpus'])
        report(status='cpu_preflight',code_sha256=hashes,arms=CONFIG['arms'],gpus=CONFIG['gpus'])
        subprocess.run([sys.executable,'-u','-m','eeg_contrastive.prepare','--out',str(output)],check=True)
        guard()
        initial = {}
        for phase in ('smoke','train'):
            require_free(CONFIG['gpus'])
            running = []
            def launch(arm,gpu):
                guard()
                require_free([gpu])
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
                directory=output/phase/arm
                command=[sys.executable,'-u','-m','eeg_contrastive.train','--arm',arm,
                         '--out',str(directory),'--bank-root',str(output)]
                if phase=='smoke': command.append('--smoke')
                with (output/f'{phase}_{arm}.log').open('x') as log:
                    return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                            env=env,start_new_session=True)
            run_on_gpu_slots(CONFIG['arms'],CONFIG['gpus'],launch,
                lambda **values:report(status=phase,**values),running=running)
            guard()
            proofs = {a:json.loads((output/phase/a/'initial_state.json').read_text()) for a in CONFIG['arms']}
            keys=('encoder_sha256','readout_sha256','projection_sha256','text_sha256')
            for key in keys:
                if len({p[key] for p in proofs.values()})!=1:
                    raise RuntimeError(f'Unmatched initialization: {key}')
            sequences = []
            for arm in CONFIG['arms']:
                done=json.loads((output/phase/arm/'complete.json').read_text())
                if done['test_evaluations'] or not all(done[k] for k in ('checkpoint_reload_verified','frozen_text_verified','encoder_updated')):
                    raise RuntimeError('Incomplete smoke/train gate')
                records=[json.loads(line) for line in (output/phase/arm/'metrics.jsonl').read_text().splitlines()]
                sequences.append([(r['step'],r['batch_ids_sha256']) for r in records if r['status']=='training'])
                if phase=='train' and any(proofs[arm][k]!=initial[arm][k] for k in keys):
                    raise RuntimeError('Smoke parameters leaked into formal training')
            if sequences[0]!=sequences[1]:
                raise RuntimeError('Arms saw different training samples')
            if phase=='smoke': initial=proofs
            report(status=phase+'_complete',initialization_matched=True,sampling_matched=True)
        report(status='complete',selection='Compare validation BAcc; losses have different numerical scales')
    except BaseException as exc:
        report(status='failed',error=f'{type(exc).__name__}: {exc}',
               remaining_processes={a:p.pid for a,_,p in running if p.poll() is None})
        raise


if __name__=='__main__':
    main()
