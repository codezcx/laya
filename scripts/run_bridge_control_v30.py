"""Run the predeclared signal control after the live Qwen job finishes."""
import json
import os
from pathlib import Path
import subprocess
import time
from eeg_align.cache_multitask_v30 import digest

ROOT = Path('artifacts/multitask_bridge_v30')


def main():
    status = ROOT/'control_controller.json'
    if status.exists(): raise FileExistsError(status)
    def record(stage, **extra):
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra),indent=2))
        tmp.replace(status)
    try:
        while True:
            state = json.loads((ROOT/'controller.json').read_text())
            if state['stage']=='failed': raise RuntimeError('Qwen controller failed; no control launched')
            if state['stage']=='complete': break
            pid = state['controller_pid']
            proc = Path(f'/proc/{pid}')
            if not proc.exists() or b'scripts/run_bridge_v30.py' not in (proc/'cmdline').read_bytes():
                raise RuntimeError('Qwen controller no longer live without completion')
            record('waiting_qwen',qwen_controller_pid=pid)
            time.sleep(15)
        qwen = ROOT/'qwen_frozen'
        run = json.loads((qwen/'run.json').read_text())
        done = json.loads((qwen/'complete.json').read_text())
        assert digest(qwen/'best.pt') == done['best_sha256']
        assert digest('eeg_align/train_multitask_v30.py') == run['trainer_sha256']
        assert digest(ROOT/'source_cache/verified.json') == run['cache_audit_sha256']
        with (ROOT/'control_training.log').open('w') as log:
            p = subprocess.Popen(['laya/.venv-eeg/bin/python','-m','torch.distributed.run',
                '--standalone','--nproc_per_node=4','-m','eeg_align.train_multitask_v30',
                '--arm','signal','--steps',str(run['steps']),'--output',str(ROOT/'signal_control')],
                env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT)
            record('training',training_pid=p.pid)
            code = p.wait()
            if code: raise RuntimeError(f'Control trainer exited {code}; no automatic restart')
        assert (ROOT/'signal_control/complete.json').exists()
        record('complete')
    except Exception as exc:
        record('failed',error=repr(exc)); raise


if __name__=='__main__': main()
