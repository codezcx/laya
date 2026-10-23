"""Read-only task gradient comparison at a selected joint contrast checkpoint."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from eeg_align.joint_data import read_rows,Sampler,raw,OFFSETS
from eeg_align.semantic_alignment import multipositive_loss

def main():
    torch.set_num_threads(3);root=Path('artifacts/mi_emotion/contrast_full_mi')
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    model=AutoModel.from_pretrained(saved['config']['encoder']['reve_checkpoint'],trust_remote_code=True,local_files_only=True).cuda().train()
    model.load_state_dict(saved['encoder']);model.cls_query_token.requires_grad_(False)
    text=torch.load(root/'text_prototypes.pt',map_location='cpu',weights_only=False)['vectors'].cuda()
    head=torch.nn.Linear(512,text.shape[1]).cuda();head.load_state_dict(saved['head']);del saved
    rows=read_rows('train');sampler=Sampler(rows);params=[p for p in model.parameters() if p.requires_grad]+list(head.parameters());results=[]
    for step in [200,800,1200]:
        torch.manual_seed(step);np.random.seed(step);grads=[];losses=[]
        rr=[rows[i] for i in sampler.full_mi_batch(step)]
        for task in range(3):
            representations=[];labels=[]
            selected=[r for r in rr if r['task']==task]
            for r in selected:
                x,p=raw(r)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=head(model(torch.from_numpy(x)[None].cuda(),torch.from_numpy(p)[None].cuda()).mean((1,2))).float()
                representations.append(z);labels.append(r['label']+OFFSETS[task])
            loss=multipositive_loss(torch.cat(representations),torch.tensor(labels,device='cuda'),text,distributed=False)
            g=torch.autograd.grad(loss,params);flat=torch.cat([v.detach().float().flatten() for v in g]);grads.append(flat.cpu());losses.append(float(loss.detach()));del g,flat,representations,loss,z
        g=torch.stack(grads).double();norms=g.norm(dim=1);cos=(g@ g.T)/(norms[:,None]*norms[None,:]);weights=torch.tensor([.5,.25,.25]);weighted=norms*weights
        out=dict(batch_step=step,losses=losses,gradient_norms=norms.tolist(),weighted_gradient_norms=weighted.tolist(),gradient_cosines=cos.tolist(),note='Final-checkpoint train-mode gradients on three fixed batches; descriptive, not causal ablation.')
        results.append(out);print(json.dumps(out),flush=True)
    Path('artifacts/mi_emotion/diagnosis/gradients.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
