"""Source-only tiny-set fitting audit for the actual JEV EEG interface."""
import argparse,json,math,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import TokenCache
from .cache import ROOT,digest
from .jev import load_jev,JEVBridge,PREFIX,SUFFIX

OUT=Path('artifacts/jev_interface_diagnostic')
VARIANTS=('original','near','scaled','near_scaled')

class MatchTextRMS(nn.Module):
    def __init__(self,connector,rms):
        super().__init__();self.connector=connector
        self.register_buffer('rms',torch.as_tensor(rms).detach().float().clone())
    def forward(self,x):
        z=self.connector(x)
        return F.normalize(z,dim=-1)*math.sqrt(z.shape[-1])*self.rms

def configure(model,tokenizer,variant):
    assert variant in VARIANTS
    if variant.startswith('near'):
        question_options=SUFFIX.split('[question] ',1)[1].rsplit('\n[decision]:',1)[0]
        prefix='[kind] choice\n[question] '+question_options+'\n[state] EEG recording begins:\n'
        suffix='\nEEG recording ends.\n[decision]:'
        model.prefix_ids=torch.tensor(tokenizer.encode(prefix,add_special_tokens=False),device=model.head_weight.device)
        model.suffix_ids=torch.tensor(tokenizer.encode(suffix,add_special_tokens=False),device=model.head_weight.device)
    else:prefix,suffix=PREFIX,SUFFIX
    with torch.no_grad():
        reference=model.text.get_input_embeddings()(torch.cat([model.prefix_ids,model.suffix_ids])).float()
        rms=reference.square().mean().sqrt()
    if 'scaled' in variant:model.connector=MatchTextRMS(model.connector,rms)
    return dict(prefix=prefix,suffix=suffix,prefix_tokens=len(model.prefix_ids),suffix_tokens=len(model.suffix_ids),
                reference_text_rms=rms.item(),variant=variant)

def fit_metrics(logits,labels):
    pred=logits.argmax(1);result={}
    for task,classes in [('MI',[3,4]),('emotion',[0,1,2])]:
        result[task]=float(np.mean([np.mean(pred[labels==c]==c) for c in classes]))
    result['accuracy']=float(np.mean(pred==labels))
    result['prediction_counts']=np.bincount(pred,minlength=5).tolist()
    return result

@torch.no_grad()
def evaluate(model,x,mask,y,order=None):
    model.eval();logits=[]
    for i in range(0,len(y),6):
        ids=torch.arange(i,min(i+6,len(y)),device=x.device) if order is None else order[i:i+6]
        logits.append(model(x[ids],mask[ids])['logits'].float().cpu())
    z=torch.cat(logits);result=fit_metrics(z.numpy(),y.cpu().numpy())
    result['loss']=F.cross_entropy(z,y.cpu()).item()
    return result,z.numpy()

def run(variant):
    folder=OUT/variant;folder.mkdir(exist_ok=False)
    spec=json.loads((OUT/'design.json').read_text());torch.set_num_threads(4)
    cache=TokenCache();ids=spec['indices'];assert all(cache.rows[i]['split']=='train' for i in ids)
    assert digest(ROOT/'rows.jsonl')==spec['manifest_sha256']
    for path,sha in spec['code_hashes'].items():assert digest(path)==sha
    text,w,b,tok=load_jev();torch.manual_seed(42)
    model=JEVBridge(text,w,b,tok,False);description=configure(model,tok,variant)
    x,mask,y=cache.batch(ids,'cuda');del cache
    assert all(not p.requires_grad for p in model.text.parameters())
    opt=torch.optim.AdamW(model.connector.parameters(),lr=1e-4,weight_decay=.01)
    with torch.no_grad():description['initial_projected_rms']=model.connector(x)[mask].square().mean().sqrt().item()
    (folder/'config.json').write_text(json.dumps(dict(**description,design_sha256=digest(OUT/'design.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    began=time.monotonic();history=[]
    with (folder/'metrics.jsonl').open('w',buffering=1) as log:
        initial,z=evaluate(model,x,mask,y);log.write(json.dumps(dict(step=0,fit=initial))+'\n')
        for step in range(1,201):
            model.train();opt.zero_grad(set_to_none=True)
            # Same balanced twelve examples in all arms; cycle five replicates per group.
            ii=torch.tensor([g*5+(step-1)%5 for g in range(12)],device='cuda');total=0.
            for part in ii.split(6):
                loss=model(x[part],mask[part],y[part])['loss']/2
                assert torch.isfinite(loss);loss.backward();total+=loss.item()
            norm=nn.utils.clip_grad_norm_(model.connector.parameters(),1.);assert torch.isfinite(norm)
            opt.step()
            if step==1 or step%10==0:
                item=dict(step=step,batch_loss=total,gradient_norm=float(norm),elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
            if step%25==0:
                result,z=evaluate(model,x,mask,y);history.append(dict(step=step,fit=result))
                log.write(json.dumps(history[-1])+'\n');print(json.dumps(history[-1]),flush=True)
        np.save(folder/'fit_logits.npy',z)
        # Preserve task, token count, and class frequencies; break trial/label pairing.
        order=torch.arange(len(y),device=y.device);rng=np.random.default_rng(123)
        for task in [False,True]:
            jj=torch.where((y>=3)==task)[0];order[jj]=jj[torch.tensor(rng.permutation(len(jj)),device=y.device)]
        shuffled,sz=evaluate(model,x,mask,y,order);np.save(folder/'shuffled_logits.npy',sz)
        constant=x.clone()
        for task in [False,True]:
            jj=torch.where((y>=3)==task)[0];constant[jj]=x[jj].mean(0)
        means,mz=evaluate(model,constant,mask,y);np.save(folder/'task_mean_logits.npy',mz)
        with torch.no_grad():
            projected=model.connector(x);projected_rms=projected[mask].square().mean().sqrt().item()
        torch.save(model.connector.state_dict(),folder/'connector.pt')
        final=dict(final_fit=history[-1],shuffled_within_task=shuffled,task_mean=means,
            projected_rms=projected_rms,reference_text_rms=description['reference_text_rms'],
            fit_gate_passed=all(history[-1]['fit'][k]>=.95 for k in ['MI','emotion']),
            note='Training-set diagnostic only. Passing does not establish generalization or zero-shot.',
            elapsed_seconds=time.monotonic()-began)
        (folder/'complete.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);args=p.parse_args();run(args.variant)
