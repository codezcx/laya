"""Fixed 3000-step source-only budget check; no target evaluation."""
import json
import os
from pathlib import Path
import subprocess
import time
from eeg_align.cache_multitask_v30 import digest

ROOT = Path('artifacts/multitask_bridge_v31')


def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    trainer = Path('eeg_align/train_multitask_v30.py')
    old = Path('artifacts/multitask_bridge_v30')
    oldrun = json.loads((old/'qwen_frozen/run.json').read_text())
    assert digest(trainer)==oldrun['trainer_sha256']
    assert digest(old/'source_cache/verified.json')==oldrun['cache_audit_sha256']
    plan = dict(steps=3000,world_size=4,initialization='from original checkpoints and same seed, not resume',
        trainer_sha256=digest(trainer),cache_audit_sha256=digest(old/'source_cache/verified.json'),
        arms=['qwen','signal'],external_evaluation=False,
        rationale='Both v30 arms selected terminal step 500; source-only convergence audit')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    def status(stage,**extra):
        temp=ROOT/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra),indent=2));temp.replace(ROOT/'status.json')
    try:
        for arm in plan['arms']:
            assert digest(trainer)==plan['trainer_sha256']
            with (ROOT/f'{arm}.log').open('w') as log:
                p=subprocess.Popen(['laya/.venv-eeg/bin/python','-m','torch.distributed.run','--standalone',
                    '--nproc_per_node=4','-m','eeg_align.train_multitask_v30','--steps','3000',
                    '--arm',arm,'--output',str(ROOT/arm)],env=dict(os.environ,PYTHONPATH='.'),
                    stdout=log,stderr=subprocess.STDOUT)
                status('training',arm=arm,training_pid=p.pid)
                rc=p.wait()
                if rc:raise RuntimeError(f'{arm} exited {rc}; no automatic restart')
            done=json.loads((ROOT/arm/'complete.json').read_text())
            assert done['steps']==3000 and done['best_sha256']==digest(ROOT/arm/'best.pt')
        status('complete')
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
