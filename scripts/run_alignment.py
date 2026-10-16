"""Four independent source runs; freeze all selections before target evaluation."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from eeg_align.cache import ROOT,digest

OUT=Path('artifacts/eeg_text_alignment/runs')

def main():
    bank=json.loads(Path('artifacts/eeg_text_alignment/embedding_bank/complete.json').read_text())
    assert bank['reports']['last']['heldout_wording_accuracy']==1.0
    OUT.mkdir(parents=True,exist_ok=False)
    status=OUT/'status.json'
    def update(phase,**kw):
        status.write_text(json.dumps(dict(phase=phase,**kw),indent=2))
    # Verify cached feature contents once before the four readers start.
    update('verifying_source_cache')
    for shard in range(4):
        root=ROOT/str(shard);audit=json.loads((root/'complete.json').read_text())
        assert audit['manifest_sha256']==digest(ROOT/'rows.jsonl')
        for name,sha in audit['hashes'].items():assert digest(root/name)==sha
    arms=[('contrastive',42),('contrastive',43),('ce',42),('ce',43)];jobs=[]
    for gpu,(control,seed) in enumerate(arms):
        name=f'{control}_seed{seed}';log=(OUT/f'{name}.log').open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
        cmd=[sys.executable,'-m','eeg_align.train','--control',control,'--seed',str(seed),'--out',str(OUT/name),'--steps','2000']
        jobs.append((name,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    update('source_training',pids={n:p.pid for n,p,l in jobs})
    while any(p.poll() is None for n,p,l in jobs):
        failures={n:p.returncode for n,p,l in jobs if p.poll() not in [None,0]}
        if failures:
            for n,p,l in jobs:
                if p.poll() is None:p.terminate()
            update('failed',failures=failures);raise RuntimeError(failures)
        time.sleep(10)
    assert all(p.returncode==0 for n,p,l in jobs)
    freeze=OUT/'source_freeze.json'
    freeze.write_text(json.dumps(dict(runs={n:digest(OUT/n/'best.pt') for n,p,l in jobs},
        selection='source task-balanced subject balanced accuracy; targets unused for selection',
        timestamp=time.time()),indent=2))
    update('external_evaluation')
    for gpu,(name,_,log) in enumerate(jobs):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
        subprocess.run([sys.executable,'-m','eeg_align.evaluate','--run',str(OUT/name),'--freeze',str(freeze)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    update('complete')

if __name__=='__main__':main()
