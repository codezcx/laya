"""Launch the preregistered interface diagnostics, without touching any targets."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from eeg_align.data import TokenCache
from eeg_align.cache import ROOT,digest
from eeg_align.interface_diagnostic import OUT,VARIANTS

def main():
    OUT.mkdir(exist_ok=False);cache=TokenCache();groups={}
    for i,r in enumerate(cache.rows):
        if r['split']=='train':groups.setdefault((r['dataset'],r['class_id']),[]).append(i)
    rng=np.random.default_rng(20261017)
    indices=[int(i) for key in sorted(groups) for i in rng.choice(groups[key],5,replace=False)]
    assert len(groups)==12 and len(set(indices))==60
    spec=dict(indices=indices,groups=[list(k) for k in sorted(groups)],seed=20261017,steps=200,
        manifest_sha256=digest(ROOT/'rows.jsonl'),variants=VARIANTS,
        code_hashes={p:digest(p) for p in ['eeg_align/interface_diagnostic.py','eeg_align/jev.py','eeg_align/data.py','scripts/run_interface_diagnostic.py']},
        statement='Only source training examples; no validation/target selection; >=95% fit per task is an engineering gate, not research success.')
    (OUT/'design.json').write_text(json.dumps(spec,indent=2));del cache
    jobs=[]
    for gpu,name in enumerate(VARIANTS):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
        log=(OUT/f'{name}.log').open('w')
        p=subprocess.Popen([sys.executable,'-m','eeg_align.interface_diagnostic','--variant',name],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,p,log))
    status=OUT/'status.json';status.write_text(json.dumps(dict(phase='running',pids={n:p.pid for n,p,l in jobs}),indent=2))
    while True:
        codes={n:p.poll() for n,p,l in jobs};bad={n:c for n,c in codes.items() if c not in [None,0]}
        if bad:
            for n,p,l in jobs:
                if p.poll() is None:p.terminate()
            status.write_text(json.dumps(dict(phase='failed',returncodes=codes),indent=2));raise RuntimeError(bad)
        if all(c==0 for c in codes.values()):break
        time.sleep(10)
    status.write_text(json.dumps(dict(phase='complete',returncodes=codes),indent=2))

if __name__=='__main__':main()
