"""Fit from train/validation only; persist selection before independent evaluation."""
import argparse,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import config,write_json
from .model import PrototypeClassifier,masked_loss,CLASS_TASK,task_mask

def load_data():
    root=Path(config()['output']);rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    x=np.zeros((len(rows),1536),dtype='float32');seen=[]
    for i in range(4):
        assert (root/'features'/f'{i}.done.json').exists()
        d=np.load(root/'features'/f'{i}.npz');x[d['indices']]=d['features'];seen.extend(d['indices'])
    assert sorted(seen)==list(range(len(rows))) and np.isfinite(x).all()
    return rows,torch.from_numpy(x).cuda()

@torch.no_grad()
def predict(model,x):
    model.eval();return torch.cat([model(z).float() for z in x.split(512)]).cpu()

def metrics(rows,logits):
    labels=torch.tensor([r['class_id'] for r in rows]);tasks=torch.tensor([r['task'] for r in rows])
    pred=logits.masked_fill(~task_mask(tasks),-1e4).argmax(-1)
    result={'n':len(rows),'accuracy':float((pred==labels).float().mean()),'tasks':{}}
    subject_values=[];dataset_values=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['dataset']==dataset];yy=labels[ids];pp=pred[ids]
        by_subject={}
        for subject in sorted({rows[i]['subject'] for i in ids}):
            si=[i for i in ids if rows[i]['subject']==subject]
            by_subject[subject]=float((pred[si]==labels[si]).float().mean())
        acc=np.array(list(by_subject.values()));boot=np.random.default_rng(2026).choice(acc,(3000,len(acc)),replace=True).mean(1)
        vals={'n':len(ids),'accuracy':float((yy==pp).float().mean()),'balanced_accuracy':float(np.mean([float((pp[yy==c]==c).float().mean()) for c in yy.unique()])), 'subject_macro_accuracy':float(acc.mean()),'subject_95ci':np.quantile(boot,[.025,.975]).tolist(),'per_subject':by_subject}
        result['tasks'][dataset]=vals;subject_values.extend(acc);dataset_values.append(acc.mean())
    result['macro_dataset_accuracy']=float(np.mean(dataset_values))
    task_scores=[]
    for t in sorted(set(r['task'] for r in rows)):
        names={r['dataset'] for r in rows if r['task']==t}
        task_scores.append(np.mean([result['tasks'][n]['subject_macro_accuracy'] for n in names]))
    result['macro_task_accuracy']=float(np.mean(task_scores))
    # No task labels used in this diagnostic argmax; scores are NOT calibrated across tasks.
    joint=logits[:, :5].argmax(-1);result['uncalibrated_joint_accuracy']=float((joint==labels).float().mean());result['uncalibrated_task_accuracy']=float((CLASS_TASK[joint]==tasks).float().mean())
    return result

def main(arm):
    cfg=config();root=Path(cfg['output']);out=root/arm;out.mkdir(exist_ok=True)
    if (out/'run.json').exists():raise FileExistsError(out)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    rows,x=load_data();proto=torch.from_numpy(np.load(root/'text_prototypes.npy')).cuda()
    if cfg.get('center_text_prototypes',False):
        # Remove common instruction/background direction within each known task.
        proto=proto.clone()
        for lo,hi in [(0,3),(3,5),(5,7)]:proto[lo:hi]=F.normalize(proto[lo:hi]-proto[lo:hi].mean(0,keepdim=True),dim=-1)
    if arm=='random_ce':proto=F.normalize(torch.randn_like(proto),dim=-1)
    # Goal steering: arithmetic prepared only; active tasks are emotion and MI.
    train=[i for i,r in enumerate(rows) if r['split']=='train' and r['task'] in (0,1)]
    val=[i for i,r in enumerate(rows) if r['split']=='validation' and r['task'] in (0,1)]
    labels=torch.tensor([r['class_id'] for r in rows],device='cuda');tasks=torch.tensor([r['task'] for r in rows],device='cuda')
    counts=Counter((rows[i]['task'],rows[i]['dataset'],rows[i]['class_id']) for i in train)
    # Balance tasks, then datasets within tasks, then classes.
    ds_count={t:len({r['dataset'] for r in rows if r['task']==t and r['split']=='train'}) for t in (0,1)}
    class_count={0:3,1:2}
    weights=np.array([1/(counts[(rows[i]['task'],rows[i]['dataset'],rows[i]['class_id'])]*ds_count[rows[i]['task']]*class_count[rows[i]['task']]) for i in train]);weights/=weights.sum()
    torch.manual_seed(cfg['seed']);model=PrototypeClassifier(proto,arm).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    write_json(out/'run.json',{'arm':arm,'config':cfg,'train_n':len(train),'val_n':len(val),'active_tasks':[0,1],'text_prototype_note':'Qwen is used offline only; no decoder in EEG inference'})
    best=-1;start=time.monotonic()
    for step in range(1,cfg['steps']+1):
        model.train();ids=rng.choice(train,cfg['batch_size'],replace=True,p=weights)
        opt.zero_grad(set_to_none=True);logits=model(x[ids]);loss=masked_loss(logits,labels[ids],tasks[ids],'sigmoid' if arm=='text_sigmoid' else 'ce')
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step==1 or step%50==0:
            record={'step':step,'loss':loss.item(),'seconds':time.monotonic()-start}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%250==0 or step==cfg['steps']:
            result=metrics([rows[i] for i in val],predict(model,x[val]));result['step']=step
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print('validation',result['macro_dataset_accuracy'],flush=True)
            if result['macro_task_accuracy']>best:
                best=result['macro_task_accuracy'];torch.save(model.state_dict(),out/'best.pt');write_json(out/'best_metrics.json',result)
    write_json(out/'frozen.json',{'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'seconds':time.monotonic()-start})
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    # External Lee and source test are read only after source selection is persisted.
    results={}
    for split in ('test','external_lee'):
        ids=[i for i,r in enumerate(rows) if r['split']==split and r['task'] in (0,1)];selected=[rows[i] for i in ids]
        logits=predict(model,x[ids]);results[split]=metrics(selected,logits);np.save(out/f'{split}_logits.npy',logits.numpy())
        shuffled=np.random.default_rng(723).permutation(ids);results[split+'_shuffle']=metrics(selected,predict(model,x[shuffled]))
    write_json(out/'evaluation.json',results);write_json(out/'complete.json',{'seconds':time.monotonic()-start,'steps':cfg['steps']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['text_ce','text_sigmoid','random_ce','learned_ce']);main(p.parse_args().arm)
