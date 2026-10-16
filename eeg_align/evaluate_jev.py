"""Source-selected actual JEV evaluation; never use target scores for selection."""
import argparse,json,pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.io import loadmat
from .cache import digest
from .data import TokenCache
from .evaluate import metrics
from .jev import load_jev,JEVBridge
from .signals import SignalCache,SignalEncoder
from .train_jev import validate
from eeg_tasks.data import preprocess,normalized_window
from eeg_tasks.normalization import scale_window
from eeg_tasks.channels import picks
from eeg_tasks.emotion_labels import SOURCE


def raw_external(rows):
    """Same preprocessing/order as the existing external token cache."""
    signals=np.zeros((len(rows),30,600),np.float32);channels=np.zeros(len(rows),np.int64)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['path']].append(i)
    names=[s.split()[-1] for s in (SOURCE/'channel_62_pos.locs').read_text().splitlines() if s.strip()]
    cp=picks(names)
    for path,ids in groups.items():
        if rows[ids[0]]['dataset']=='2b':
            assert len(ids)==1
            with open(path,'rb') as f:item=pickle.load(f)
            assert item['freq']==200
            x=np.stack([item['eeg'][c] for c in ['C3','CZ','C4']])
            x=preprocess(x,200,{'bandpass':[1,30]})[:,:600]
            signals[ids[0],:3]=scale_window(x,'common_scale');channels[ids]=3
        else:
            rr=[rows[i] for i in ids];assert all(r['dataset']=='SEED-VII-HNS' for r in rr)
            data=loadmat(path,variable_names=list({r['trial'] for r in rr}));filtered={}
            for i,r in zip(ids,rr):
                trial=r['trial']
                if trial not in filtered:filtered[trial]=preprocess(data[trial][cp],200,{'bandpass':[1,30]})
                start=r['start_sample'];signals[i]=normalized_window(filtered[trial][:,start:start+600]);channels[i]=30
    assert np.isfinite(signals).all() and (channels>0).all()
    return signals,channels

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    freeze=json.loads(args.freeze.read_text());assert freeze['runs'][args.run.name]==digest(args.run/'best.pt')
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False);cfg=saved['config']
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,cfg['train_lora'])
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(saved['trainable'])==expected
    model.load_state_dict(saved['trainable'],strict=False);model.eval()
    encoder=None;raw=None
    if cfg['train_encoder']:
        encoder=SignalEncoder(trainable=False).eval();encoder.load_state_dict(saved['encoder']);raw=SignalCache()
    cache=TokenCache();ids=[i for i,r in enumerate(cache.rows) if r['split']=='validation']
    full=validate(model,cache,raw,encoder,ids)
    (args.run/'full_source_validation.json').write_text(json.dumps(full,indent=2));del cache,raw
    rows=[];scores=[];provenance={}
    for shard in range(4):
        root=Path('artifacts/multitask_bridge_v30/external')/str(shard);audit=json.loads((root/'complete.json').read_text())
        for name,sha in audit['hashes'].items():assert digest(root/name)==sha
        provenance[str(root)]=digest(root/'complete.json')
        rr=[json.loads(s) for s in (root/'rows.jsonl').read_text().splitlines()]
        tokens=np.load(root/'tokens.npy',mmap_mode='r');lengths=np.load(root/'lengths.npy')
        if encoder is not None:signals,channels=raw_external(rr)
        for start in range(0,len(rr),12):
            if encoder is None:
                ll=lengths[start:start+12];k=int(ll.max());z=torch.tensor(np.array(tokens[start:start+12,:k]),device='cuda',dtype=torch.float32)
                m=torch.arange(k,device='cuda')[None]<torch.tensor(ll,device='cuda')[:,None]
            else:
                x=torch.tensor(signals[start:start+12],device='cuda');ch=torch.tensor(channels[start:start+12],device='cuda')
                z,m=encoder(x,ch)
            scores.append(model(z,m)['logits'].cpu().numpy())
        rows.extend(rr);print('external shard complete',shard,flush=True)
    logits=np.concatenate(scores);np.save(args.run/'external_logits.npy',logits);pred=logits.argmax(1);result={}
    for ds in sorted({r['dataset'] for r in rows}):
        ii=[i for i,r in enumerate(rows) if r['dataset']==ds];result[ds]=metrics([rows[i] for i in ii],pred[ii])
    (args.run/'external.json').write_text(json.dumps(dict(datasets=result,source_freeze_sha256=digest(args.freeze),
        cache_provenance=provenance,note='Exploratory exposed targets; REVE pretrained on 2b; no target-based selection'),indent=2))

if __name__=='__main__':main()
