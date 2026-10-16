"""Compare real two-rank continuous and resumed smoke checkpoints before launch."""
import json
from pathlib import Path
import numpy as np
import torch
from eeg_align.cache import digest

ROOT=Path('artifacts/jev_eeg_coverage/smoke')

def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}.{i}')
    else:assert a==b,path

def main():
    for arm in ['continuous','resumed']:assert (ROOT/arm/'complete.json').exists()
    a=torch.load(ROOT/'continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(ROOT/'resumed/last.pt',map_location='cpu',weights_only=False)
    for key in ['trainable','optimizer','sampler','rng_states','best_score','step','config']:equal(a[key],b[key],key)
    for path,sha in a['config']['code_sha256'].items():assert digest(path)==sha
    result=dict(exact_match=True,step=a['step'],compared=['trainable','optimizer','sampler','rng_states','best_score','config'],
                code_sha256=a['config']['code_sha256'],checkpoints={arm:digest(ROOT/arm/'last.pt') for arm in ['continuous','resumed']})
    (ROOT/'resume_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
