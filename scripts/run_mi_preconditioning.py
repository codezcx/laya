"""Run matched CE/text-alignment -> shuffled-option JEV arms, two GPUs each."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'laya/.venv-eeg/bin/python'
ARMS=[('ce','0,1',29671),('contrast','2,3',29672)]


def main():
    os.chdir(ROOT)
    status=ROOT/'artifacts/mi_preconditioning_pipeline.json'
    state={'started':time.time(),'arms':{}}
    def write():
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
    def launch(arm,gpus,port,stage):
        out=Path(f'artifacts/mi_precondition_{arm}' if stage=='pretrain' else f'artifacts/mi_jev_precondition_{arm}')
        assert not out.exists(),out
        args=['-m','torch.distributed.run','--nproc-per-node=2',f'--master-port={port}']
        if stage=='pretrain':args+=['-m','eeg_align.train_reve','--objective',arm,'--epochs','1','--skip-test','--out',str(out)]
        else:
            source=Path(f'artifacts/mi_precondition_{arm}')
            assert (source/'complete.json').exists() and (source/'best.pt').exists()
            args+=['-m','eeg_align.train_jev','--recipe','tuab-lr','--shuffle-options','--microbatch','4',
                   '--encoder-checkpoint',str(source/'best.pt'),'--out',str(out)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false')
        with Path(str(out)+'.log').open('x') as log:
            proc=subprocess.Popen([str(PYTHON),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        state['arms'][arm]=dict(stage=stage,pid=proc.pid,gpus=gpus,out=str(out),started=time.time());write()
        return proc
    processes={arm:launch(arm,gpus,port,'pretrain') for arm,gpus,port in ARMS}
    # Advance each arm independently; fixed disjoint devices prevent contention.
    while processes:
        for arm,gpus,port in ARMS:
            if arm not in processes:continue
            rc=processes[arm].poll()
            if rc is None:continue
            stage=state['arms'][arm]['stage']
            state['arms'][arm].update(returncode=rc,finished=time.time());write()
            if rc!=0:
                state['arms'][arm]['stage']='failed_'+stage;del processes[arm];write()
            elif stage=='pretrain':processes[arm]=launch(arm,gpus,port,'jev')
            else:
                state['arms'][arm]['stage']='complete';del processes[arm];write()
        time.sleep(5)
    state['finished']=time.time();write()

if __name__=='__main__':main()
