"""Wait for existing cache workers, verify, then run one four-GPU training job."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('artifacts/multitask_bridge_v30')
PYTHON = 'laya/.venv-eeg/bin/python'


def main():
    status = ROOT/'controller.json'
    if status.exists(): raise FileExistsError(status)
    jobs = json.loads((ROOT/'source_cache/jobs.json').read_text())
    def write(stage, **kwargs):
        payload = dict(stage=stage,controller_pid=os.getpid(),updated=time.time(),**kwargs)
        temp = status.with_suffix('.tmp'); temp.write_text(json.dumps(payload,indent=2)); temp.replace(status)
    try:
        while True:
            pending = []
            for job in jobs:
                if (ROOT/f"source_cache/{job['shard']}/complete.json").exists(): continue
                proc = Path(f"/proc/{job['pid']}")
                try:
                    cmd = (proc/'cmdline').read_bytes()
                    stat = (proc/'stat').read_text().split(') ',1)[1].split()[0]
                except FileNotFoundError:
                    raise RuntimeError(f'Cache worker missing without completion: {job}')
                if b'eeg_align.cache_multitask_v30' not in cmd or stat=='Z':
                    raise RuntimeError(f'Cache worker terminal or replaced: {job}')
                pending.append(job['shard'])
            if not pending: break
            write('waiting_existing_cache',pending=pending); time.sleep(15)
        env = dict(os.environ,PYTHONPATH='.')
        write('verifying')
        with (ROOT/'verify.log').open('w') as log:
            subprocess.run([PYTHON,'-m','eeg_align.cache_multitask_v30','--verify'],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        with (ROOT/'training.log').open('w') as log:
            process = subprocess.Popen([PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                '-m','eeg_align.train_multitask_v30','--steps','500',
                '--output',str(ROOT/'qwen_frozen')],env=env,stdout=log,stderr=subprocess.STDOUT)
            write('training',training_pid=process.pid)
            code = process.wait()
            if code: raise RuntimeError(f'Trainer exited {code}; no automatic restart')
        assert (ROOT/'qwen_frozen/complete.json').exists()
        write('complete')
    except Exception as exc:
        write('failed',error=repr(exc)); raise


if __name__ == '__main__': main()
