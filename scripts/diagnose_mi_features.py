"""Fixed ridge probes on source data only; diagnostic, never target-selected."""
from collections import Counter
import numpy as np
import torch
from eeg_align.dg_pilot import ROOT, Features, metrics, write_json

torch.set_num_threads(4)
report = {"purpose":"Source-only representation diagnostic, not a target-selected model",
          "alpha":100.,"feature":"Electrode-wise temporal mean; training-only standardization"}
for mode in ("legacy", "harmonized"):
    data = Features(mode)
    tr=[i for i,r in enumerate(data.rows) if r["original_split"]=="train" and r["dataset"]!="Lee2019_MI"]
    va=[i for i,r in enumerate(data.rows) if r["original_split"]!="train" and r["dataset"]=="Lee2019_MI"]
    def arrays(ids):
        x=np.stack([data.arrays[data.rows[i]["shard"]][data.rows[i]["local_index"],:data.rows[i]["token_lengths"][mode]].reshape(3,-1,512).mean(1).ravel() for i in ids])
        y=np.array([[data.rows[i]["label"],*data.rows[i]["aux_labels"]] for i in ids])
        return torch.tensor(x,device="cuda",dtype=torch.float64),torch.tensor(y,device="cuda",dtype=torch.float64)
    x,y=arrays(tr); xv,yv=arrays(va)
    mean=x.mean(0); std=x.std(0).clamp_min(1e-6)
    x=(x-mean)/std; xv=(xv-mean)/std
    x=torch.cat([x,torch.ones(len(x),1,device="cuda",dtype=x.dtype)],1)
    xv=torch.cat([xv,torch.ones(len(xv),1,device="cuda",dtype=x.dtype)],1)
    counts=Counter(data.rows[i]["dataset"] for i in tr)
    w=torch.tensor([len(tr)/(len(counts)*counts[data.rows[i]["dataset"]]) for i in tr],device="cuda",dtype=x.dtype)
    penalty=torch.eye(x.shape[1],device="cuda",dtype=x.dtype)*100
    penalty[-1,-1]=0
    coef=torch.linalg.solve(x.T@(x*w[:,None])+penalty,x.T@((2*y-1)*w[:,None]))
    scores=(xv@coef).cpu().numpy()
    train_scores=(x@coef).cpu().numpy()
    entries={}
    for task,name in enumerate(("mi","mu_C3_vs_C4","beta_C3_vs_C4")):
        rows=[dict(data.rows[i]) for i in va]
        if task:
            for r in rows:r["label"]=r["aux_labels"][task-1]
        entries[name]=metrics(rows,np.stack([-scores[:,task],scores[:,task]],1))
        entries[name]["resubstitution_accuracy"]=float(np.mean((train_scores[:,task]>0)==y[:,task].cpu().numpy()))
    report[mode]=entries
    print({"mode":mode,**{k:{"validation":v["accuracy"],"train":v["resubstitution_accuracy"],"majority":v["majority_baseline_accuracy"]} for k,v in entries.items()}},flush=True)
write_json(ROOT/"source_ridge_diagnostic.json",report)
