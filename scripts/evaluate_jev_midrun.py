"""Read-only diagnostic of a snapshot, concurrent with training; never selects models."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from eeg_align.jev import load_jev,JEVBridge
from eeg_align.data import TokenCache,metrics as source_metrics
from eeg_align.evaluate import metrics
from eeg_align.cache import digest

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45)
    root=args.snapshot;save=torch.load(root/'snapshot.pt',map_location='cpu',weights_only=False)
    cfg=save['config'];assert not cfg['smoke']
    for path,sha in cfg['code_sha256'].items():assert digest(path)==sha
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,cfg['train_lora'])
    expected={n for n,p in model.named_parameters() if p.requires_grad};assert expected==set(save['trainable'])
    model.load_state_dict(save['trainable'],strict=False);model.eval();model.requires_grad_(False)
    step=save['step'];del save
    cache=TokenCache();groups={}
    for i,r in enumerate(cache.rows):
        if r['split']=='validation':groups.setdefault((r['dataset'],r['subject'],r['class_id']),[]).append(i)
    rng=np.random.default_rng(20261016);ids=[]
    for key,values in sorted(groups.items()):ids.extend(rng.choice(values,min(2,len(values)),replace=False).tolist())
    def evaluate(rows,logits):
        pred=logits.argmax(1);result={}
        for ds in sorted({r['dataset'] for r in rows}):
            ii=[i for i,r in enumerate(rows) if r['dataset']==ds]
            result[ds]=metrics([rows[i] for i in ii],pred[ii])
        return result
    values=[];began=time.time()
    for start in range(0,len(ids),12):
        z,m,_=cache.batch(ids[start:start+12],'cuda');values.append(model(z,m)['logits'].cpu().numpy())
    logits=np.concatenate(values);np.save(root/'source_panel_logits.npy',logits)
    panel=dict(step=step,ids=ids,balanced=source_metrics([cache.rows[i] for i in ids],logits),
        datasets=evaluate([cache.rows[i] for i in ids],logits),note='Diagnostic panel only; no checkpoint selection',seconds=time.time()-began)
    (root/'source_panel.json').write_text(json.dumps(panel,indent=2));print('SOURCE PANEL',json.dumps(panel['balanced']),flush=True)
    del cache
    rows=[];values=[]
    for shard in range(4):
        folder=Path('artifacts/multitask_bridge_v30/external')/str(shard)
        audit=json.loads((folder/'complete.json').read_text())
        for name,sha in audit['hashes'].items():assert digest(folder/name)==sha
        rr=[json.loads(s) for s in (folder/'rows.jsonl').read_text().splitlines()]
        tokens=np.load(folder/'tokens.npy',mmap_mode='r');lengths=np.load(folder/'lengths.npy')
        for start in range(0,len(rr),12):
            ll=lengths[start:start+12];k=int(ll.max())
            z=torch.tensor(np.array(tokens[start:start+12,:k]),device='cuda',dtype=torch.float32)
            mask=torch.arange(k,device='cuda')[None]<torch.tensor(ll,device='cuda')[:,None]
            values.append(model(z,mask)['logits'].cpu().numpy())
            if start%240==0:
                progress=dict(step=step,shard=shard,shard_completed=start+len(ll),elapsed_seconds=time.time()-began)
                (root/'progress.json').write_text(json.dumps(progress));print('EXTERNAL',json.dumps(progress),flush=True)
        rows.extend(rr)
    logits=np.concatenate(values);np.save(root/'external_logits.npy',logits)
    result=dict(step=step,datasets=evaluate(rows,logits),snapshot_sha256=digest(root/'snapshot.pt'),
        note='User-requested mid-training diagnostic; exposed targets; not source-selected final results',seconds=time.time()-began)
    (root/'external.json').write_text(json.dumps(result,indent=2));print('EXTERNAL COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
