"""Diagnose MI information before/after JEV using fixed source splits and matched label budgets."""
import argparse
import json
from pathlib import Path
import numpy as np
from eeg_align.cache import digest


def extract(args):
    import torch
    from eeg_align.attribute_data import AttributeData,QuestionTokens
    from eeg_align.paired_cache import PairedTokenCache
    from eeg_align.jev import load_jev,JEVBridge
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.48,torch.device('cuda:0'))
    args.output.mkdir(parents=True,exist_ok=False)
    sha=digest(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config'];assert cfg['fixed_mi_class']
    assert cfg['token_audit_sha256']==digest('artifacts/jev_paired_qa/tokens/verified.json')
    text,w,b,tok=load_jev('cuda:0');model=JEVBridge(text,w,b,tok,train_lora=False)
    assert set(saved['trainable'])==set(cfg['trainable_names'])
    assert not model.load_state_dict(saved['trainable'],strict=False).unexpected_keys
    model.requires_grad_(False).eval();step=saved['step'];del saved
    cache=PairedTokenCache();records=[];cursor=0;captured=[]
    states=np.lib.format.open_memmap(args.output/'states.npy',mode='w+',dtype='float32',shape=(9800,5120))
    reve=np.lib.format.open_memmap(args.output/'reve.npy',mode='w+',dtype='float32',shape=(9800,3072))
    logits=np.lib.format.open_memmap(args.output/'logits.npy',mode='w+',dtype='float32',shape=(9800,2))
    def hook(module,inputs,kwargs,output):
        h=output.last_hidden_state;last=kwargs['attention_mask'].long().sum(1)-1
        captured.append(h[torch.arange(len(h),device=h.device),last].float().cpu().numpy())
    handle=model.text.register_forward_hook(hook,with_kwargs=True)
    with torch.no_grad():
        for split,root,key in [('train','datasets/eeg_attribute_qa','train_data_sha256'),
                               ('validation','datasets/eeg_attribute_qa_validation','validation_data_sha256')]:
            data=AttributeData(root,split);assert data.root_hash==cfg[key]
            q=next(i for i,s in enumerate(data.specs) if s['id']=='mi_class')
            rows=data.records[data.records[:,1]==q]
            sources=[json.loads(s) for s in (Path(root)/'mi_sources.jsonl').read_text().splitlines()]
            people={r['source_row_index']:str(r['source_metadata']['subject']) for r in sources}
            questions=QuestionTokens(tok,data.specs,device='cuda:0')
            for start in range(0,len(rows),24):
                rr=rows[start:start+24];z,mask,cuts=cache.batch(rr[:,0],'cuda:0')
                suffix,_=questions.batch(rr,np.zeros(len(rr),dtype=int))
                out=model(z,mask,suffix_ids=suffix,num_choices=2,segment_splits=cuts)['logits']
                assert len(captured)==1
                state=captured.pop();assert np.isfinite(state).all()
                # Verify the captured state is exactly the state scored by the fixed head.
                rebuilt=torch.nn.functional.linear(torch.as_tensor(state,device='cuda:0'),model.head_weight[:2],model.head_bias[:2])
                torch.testing.assert_close(rebuilt,out,rtol=1e-5,atol=1e-5)
                sl=slice(cursor,cursor+len(rr));states[sl]=state;logits[sl]=out.cpu().numpy()
                reve[sl]=z[:,:18].float().reshape(len(rr),2,3,3,512).mean(3).reshape(len(rr),3072).cpu().numpy()
                records.extend(dict(source_id=int(r[0]),label=int(r[2]),subject=people[int(r[0])],split=split) for r in rr)
                cursor+=len(rr)
                if start%480==0:print('Extract decision states',split,start,len(rows),flush=True)
    handle.remove();assert cursor==9800 and sha==digest(args.checkpoint)
    for a in [states,reve,logits]:a.flush()
    (args.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    protocol=dict(step=step,checkpoint=str(args.checkpoint),checkpoint_sha256=sha,
        seed=cfg['seed'],classification_per_class=cfg['classification_per_class'],
        source_data_sha256=cfg['train_data_sha256'],validation_data_sha256=cfg['validation_data_sha256'],
        source_cache_sha256=cfg['token_audit_sha256'],code_sha256=digest(__file__),
        limitation='Probe alpha tuning uses held-out probe subjects, but the frozen JEV checkpoint was trained on all source training subjects. Source validation has been repeatedly observed; not blind or end-to-end nested validation.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (args.output/'extraction_complete.json').write_text(json.dumps({n:digest(args.output/n) for n in ['states.npy','reve.npy','logits.npy','records.jsonl','protocol.json']},indent=2))


def fit(args):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import balanced_accuracy_score
    from eeg_align.attribute_data import AttributeData
    from eeg_align.sampling import BalancedCoverageSampler
    from scripts.evaluate_paired_classification import report
    assert not (args.output/'results.json').exists()
    for n,sha in json.loads((args.output/'extraction_complete.json').read_text()).items():assert digest(args.output/n)==sha
    protocol=json.loads((args.output/'protocol.json').read_text())
    records=[json.loads(s) for s in (args.output/'records.jsonl').read_text().splitlines()]
    y=np.array([r['label'] for r in records]);subjects=np.array([r['subject'] for r in records])
    train=np.array([r['split']=='train' for r in records]);test=~train
    assert not set(subjects[train]) & set(subjects[test])
    people=np.unique(subjects[train]);np.random.default_rng(42).shuffle(people)
    inner=train & np.isin(subjects,people[:9]);training=train & ~inner
    data=AttributeData('datasets/eeg_attribute_qa','train');assert data.root_hash==protocol['source_data_sha256']
    q=next(i for i,s in enumerate(data.specs) if s['id']=='mi_class')
    counts={k:protocol['classification_per_class'] if k[0]==q else 1 for k in data.groups}
    sampler=BalancedCoverageSampler(data.groups,per_group=counts,seed=protocol['seed']);seen=set()
    for _ in range(protocol['step']):
        rr=data.records[sampler.next_batch().reshape(-1)]
        seen.update(rr[rr[:,1]==q,0].tolist())
    budgets={'seen_classification_labels':train & np.array([r['source_id'] in seen for r in records]),'all_training_labels':train}
    target=np.eye(2)[y];results={};alphas=[10,1000,100000]
    np.savez(args.output/'probe_partitions.npz',fit=training,inner=inner,test=test,seen_labels=budgets['seen_classification_labels'])
    (args.output/'readout_protocol.json').write_text(json.dumps(dict(alphas=alphas,
        fit_subjects=sorted(set(subjects[training])),inner_subjects=sorted(set(subjects[inner])),
        validation_subjects=sorted(set(subjects[test])),budgets={k:int(v.sum()) for k,v in budgets.items()},
        limitation=protocol['limitation']),indent=2))
    for name in ['reve','states']:
        x=np.load(args.output/f'{name}.npy',mmap_mode='r')
        for budget,allowed in budgets.items():
            scores=[]
            for alpha in alphas:
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
                model.fit(x[allowed & training],target[allowed & training])
                scores.append(float(balanced_accuracy_score(y[allowed & inner],model.predict(x[allowed & inner]).argmax(1))))
            alpha=alphas[int(np.argmax(scores))]
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
            model.fit(x[allowed],target[allowed]);pred=model.predict(x[test]).argmax(1)
            key=f'{name}_{budget}';results[key]=dict(alpha=alpha,inner_scores=scores,n_labels=int(allowed.sum()),
                training_accuracy=float(np.mean(model.predict(x[allowed]).argmax(1)==y[allowed])),
                validation=report(y[test],pred,subjects[test]))
            np.save(args.output/f'{key}_predictions.npy',pred)
            print(key,'accuracy',results[key]['validation']['overall']['accuracy'],flush=True)
    fixed=np.load(args.output/'logits.npy')[test].argmax(1)
    results['fixed_head']=report(y[test],fixed,subjects[test])
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    (args.output/'complete.json').write_text(json.dumps(dict(results_sha256=digest(args.output/'results.json'),code_sha256=digest(__file__)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['extract','fit'],required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.stage=='extract':
        if args.checkpoint is None:p.error('--checkpoint required for extract')
        extract(args)
    else:fit(args)
