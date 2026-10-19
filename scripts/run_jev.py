"""TUAB preparation, four-GPU REVE caching, smoke test and record classifier."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from eeg_align.tuab_data import ROOT,digest,readrows,write_json
OUT=Path('artifacts/tuab_jev')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3')
    def status(phase,**kw):write_json(OUT/'status.json',dict(phase=phase,time=time.time(),**kw))
    def run(name,command):
        with (OUT/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            status(name,pid=proc.pid,command=command);code=proc.wait()
        if code:
            status('failed',stage=name,returncode=code);raise RuntimeError(f'{name} failed: {code}')
    run('prepare',[sys.executable,'-m','eeg_align.tuab_data','--prepare'])
    procs=[];logs=[]
    for shard in range(4):
        log=(OUT/f'encode_{shard}.log').open('x');logs.append(log)
        command=[sys.executable,'-m','eeg_align.tuab_data','--shard',str(shard)]
        procs.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard))))
    status('encoding',pids=[p.pid for p in procs])
    codes=[p.wait() for p in procs]
    for f in logs:f.close()
    if any(codes):status('failed',stage='encoding',codes=codes);raise RuntimeError(codes)
    expected={r['id'] for s in ['train','validation','test'] for r in readrows(ROOT/f'records_{s}.jsonl')};seen=set()
    for shard in range(4):
        audit=json.loads((ROOT/'tokens'/f'shard_{shard}.json').read_text())
        assert audit['manifest_sha256']==digest(ROOT/'complete.json')
        for record,sha in audit['files'].items():
            assert record not in seen and digest(ROOT/'tokens'/f'{record}.npy')==sha;seen.add(record)
    assert seen==expected
    write_json(OUT/'token_audit.json',dict(status='passed',records=len(seen)))
    run('probe',[sys.executable,'scripts/tuab_probe.py'])
    for name,args in [('smoke',['--smoke']),('run',[])]:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
             '-m','eeg_align.train_jev','--out',str(OUT/name),*args]
        run(name,cmd)
        assert (OUT/name/'complete.json').exists()
    run('probe_test',[sys.executable,'scripts/tuab_probe.py','--test'])
    status('complete')


if __name__=='__main__':main()
