"""Current paired-MI / unchanged-emotion REVE cache, keyed by source identity."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .cache import digest
from .data import TokenCache

ROOT=Path('artifacts/jev_paired_qa/tokens')
DATA=[Path('datasets/eeg_attribute_qa'),Path('datasets/eeg_attribute_qa_validation')]


def sources():
    result=[]
    for root in DATA:
        for task in ['mi','emotion']:
            for line in (root/f'{task}_sources.jsonl').read_text().splitlines():
                r=json.loads(line)
                result.append((r['source_row_index'],str(root),task,r['signal_index']))
    assert len({r[0] for r in result})==len(result)
    return sorted(result)


@torch.no_grad()
def extract(shard):
    torch.set_num_threads(4)
    out=ROOT/str(shard);out.mkdir(parents=True,exist_ok=False)
    records=sources()[shard::4]
    features=np.lib.format.open_memmap(out/'tokens.npy',mode='w+',dtype=np.float16,shape=(len(records),120,512))
    features[:]=0
    lengths=np.array([18 if r[2]=='mi' else 120 for r in records])
    old=TokenCache()
    emotion=[i for i,r in enumerate(records) if r[2]=='emotion']
    for start in range(0,len(emotion),64):
        ii=emotion[start:start+64]
        z,mask,_=old.batch([records[i][0] for i in ii],'cpu')
        assert mask.all()
        features[ii]=z.numpy().astype(np.float16)
    del old
    cfg=json.loads(Path('configs/encoder.json').read_text())
    model,info=AutoModel.from_pretrained(cfg['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.cuda().eval().requires_grad_(False)
    bank=AutoModel.from_pretrained(cfg['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
    positions=bank(['C3','Cz','C4']).detach().cuda()
    arrays={str(root):[np.load(root/f'signals/mi_{s}.npy',mmap_mode='r') for s in ['baseline','task']] for root in DATA}
    mi=[i for i,r in enumerate(records) if r[2]=='mi']
    for start in range(0,len(mi),32):
        ii=mi[start:start+32]
        # Encode the two segments separately, maintaining their shared input amplitude scale.
        x=np.stack([np.stack([a[records[i][3]] for a in arrays[records[i][1]]]) for i in ii])
        x=torch.from_numpy(x.reshape(-1,3,400)).cuda()
        tail=(-(400-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap)
        x=torch.nn.functional.pad(x,(0,tail),mode='replicate')
        z=model(x,positions[None].expand(len(x),-1,-1))
        assert z.shape==(len(x),3,3,512) and torch.isfinite(z).all()
        features[ii,:18]=z.reshape(len(ii),18,512).cpu().numpy().astype(np.float16)
        if start%640==0:print('paired cache',shard,start,len(mi),flush=True)
    features.flush();del features
    np.save(out/'indices.npy',np.array([r[0] for r in records]))
    np.save(out/'lengths.npy',lengths)
    (out/'complete.json').write_text(json.dumps(dict(schema_version=2,n=len(records),
        dataset_sha256={str(p):digest(p/'complete.json') for p in DATA},
        files={f:digest(out/f) for f in ['tokens.npy','indices.npy','lengths.npy']},
        mi_layout='baseline 9 tokens then task 9 tokens; independent REVE encoding; shared amplitude normalization',
        emotion='byte-equivalent frozen REVE tokens from original verified cache',
        encoder_config=cfg,code_sha256=digest(__file__)),indent=2))
    print('cache complete',shard,len(records),flush=True)


def verify():
    expected={r[0]:r[2] for r in sources()};seen=[]
    for shard in range(4):
        folder=ROOT/str(shard);audit=json.loads((folder/'complete.json').read_text())
        assert audit['dataset_sha256']=={str(p):digest(p/'complete.json') for p in DATA}
        for f,sha in audit['files'].items():assert digest(folder/f)==sha
        ids=np.load(folder/'indices.npy');ll=np.load(folder/'lengths.npy');x=np.load(folder/'tokens.npy',mmap_mode='r')
        assert x.shape==(len(ids),120,512)
        for k,i in enumerate(ids):
            assert ll[k]==(18 if expected[int(i)]=='mi' else 120)
            assert np.isfinite(x[k]).all() and np.all(x[k,ll[k]:]==0)
        seen.extend(ids.tolist())
    assert len(seen)==len(set(seen)) and set(seen)==set(expected)
    (ROOT/'verified.json').write_text(json.dumps(dict(n=len(seen),dataset_sha256={str(p):digest(p/'complete.json') for p in DATA},
        shard_sha256={str(s):digest(ROOT/str(s)/'complete.json') for s in range(4)}),indent=2))


class PairedTokenCache:
    def __init__(self):
        audit=json.loads((ROOT/'verified.json').read_text())
        assert audit['dataset_sha256']=={str(p):digest(p/'complete.json') for p in DATA}
        self.arrays=[];self.lengths=[];self.location={}
        for shard in range(4):
            folder=ROOT/str(shard)
            assert audit['shard_sha256'][str(shard)]==digest(folder/'complete.json')
            self.arrays.append(np.load(folder/'tokens.npy',mmap_mode='r'))
            self.lengths.append(np.load(folder/'lengths.npy'))
            for j,i in enumerate(np.load(folder/'indices.npy')):self.location[int(i)]=(shard,j)
    def batch(self,ids,device):
        loc=[self.location[int(i)] for i in ids]
        lengths=[int(self.lengths[s][j]) for s,j in loc];n=max(lengths)
        x=np.stack([self.arrays[s][j,:n] for s,j in loc]).astype(np.float32)
        mask=torch.arange(n,device=device)[None]<torch.tensor(lengths,device=device)[:,None]
        splits=[9 if n==18 else 0 for n in lengths]
        return torch.from_numpy(x).to(device),mask,splits


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:verify()
    else:extract(a.shard)
