"""Read-only emotion QA audit: independent labels and matched-budget frozen-REVE probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from eeg_align.attribute_data import AttributeData, training_weights
from eeg_align.sampling import BalancedCoverageSampler
from eeg_align.paired_cache import PairedTokenCache
from eeg_align.cache import digest
from eeg_tasks.channels import CHANNELS
from eeg_tasks.attribute_questions import FRONTAL, POSTERIOR
from scripts.verify_attribute_dataset import fft_power


def metrics(y,p):
    return dict(n=len(y),accuracy=float(np.mean(y==p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),
                confusion=confusion_matrix(y,p,labels=[0,1,2]).tolist(),
                prediction_counts=np.bincount(p,minlength=3).tolist())


def independent_values(x,specs):
    powers={}
    for window,sl in [('full',slice(None)),('early',slice(0,300)),('late',slice(300,600))]:
        psd=fft_power(x[...,sl]);den=psd[...,1:31].sum(-1)
        if not (den>0).all():raise ValueError('Nonpositive EEG power')
        powers[window]={band:psd[...,slc].sum(-1)/den for band,slc in [('alpha',slice(8,13)),('beta',slice(13,31))]}
    values=[]
    for spec in specs:
        vv=[]
        for side in [spec['first'],spec['second']]:
            picks=[CHANNELS.index(c) for c in side['channels']]
            vv.append(powers[side['window']][side['band']][:,picks].mean(1))
        values.append(10*np.log10(np.maximum(vv[0],1e-12)/np.maximum(vv[1],1e-12)))
    return np.stack(values,axis=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('artifacts/jev_paired_qa/emotion_diagnostic'))
    ap.add_argument('--step',type=int,default=200)
    args=ap.parse_args();out=args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    config_path=Path('artifacts/jev_paired_qa/run/config.json');cfg=json.loads(config_path.read_text())
    data=[AttributeData('datasets/eeg_attribute_qa','train',True),AttributeData('datasets/eeg_attribute_qa_validation','validation',True)]
    specs=[s for s in data[0].specs if s['task']=='emotion' and s['id']!='emotion_class']
    qs=[next(q for q,s in enumerate(data[0].specs) if s['id']==spec['id']) for spec in specs]
    sources=[];targets=[];features=[];audit={};sample_rows=[]
    cache=PairedTokenCache()
    for d in data:
        src=[json.loads(l) for l in (d.root/'emotion_sources.jsonl').read_text().splitlines()]
        x=np.load(d.root/'signals/emotion.npy',mmap_mode='r')
        assert x.shape==(len(src),30,600)
        ids=np.array([s['source_row_index'] for s in src]);lookup={int(i):j for j,i in enumerate(ids)}
        yy=np.full((len(src),len(specs)),-1,int)
        for j,q in enumerate(qs):
            for i,_,a,_ in d.records[d.records[:,1]==q]: yy[lookup[int(i)],j]=a
        assert (yy>=0).all()
        xx=np.empty((len(src),4096),np.float32);mismatches=np.zeros(len(specs),int);max_error=0.
        for begin in range(0,len(src),128):
            end=min(begin+128,len(src));v=independent_values(np.asarray(x[begin:end]),specs)
            expected=np.where(v>1,0,np.where(v < -1,2,1))
            mismatches+=(expected!=yy[begin:end]).sum(0)
            z,mask,cuts=cache.batch(ids[begin:end].tolist(),'cpu')
            assert mask.all() and not any(cuts) and z.shape[1:]==(120,512)
            z=z.numpy().reshape(-1,30,4,512)
            xx[begin:end]=np.stack([z[:,[CHANNELS.index(c) for c in region]].mean(1) for region in [FRONTAL,POSTERIOR]],axis=1).reshape(end-begin,-1)
        assert not mismatches.any(),('Independent labels differ',mismatches.tolist())
        audit[str(d.root)]=dict(n=len(src),checked_answers=int(yy.size),mismatches=mismatches.tolist(),
            signals_sha256=digest(d.root/'signals/emotion.npy'),question_hashes={s['id']:digest(d.root/'questions'/f"{s['id']}.jsonl") for s in specs})
        sources.extend(src);targets.append(yy);features.append(xx)
        print('Independent labels and REVE features verified',d.root,len(src),flush=True)
    x=np.concatenate(features).astype(np.float64);y=np.concatenate(targets);del features
    ids=np.array([s['source_row_index'] for s in sources]);lookup={int(i):j for j,i in enumerate(ids)}
    subject=np.array([str(s['source_metadata']['dataset'])+':'+str(s['source_metadata']['subject']) for s in sources])
    dataset=np.array([s['source_metadata']['dataset'] for s in sources]);train=np.arange(len(sources))<len(targets[0]);test=~train
    assert not set(subject[train]) & set(subject[test])
    inner_subjects=[];rng=np.random.default_rng(42)
    for ds in sorted(set(dataset)):
        people=np.unique(subject[train & (dataset==ds)]);rng.shuffle(people);inner_subjects.extend(people[:max(1,len(people)//4)])
    inner=np.isin(subject,inner_subjects)&train
    d=data[0];counts={k:cfg['classification_per_class'] if d.specs[k[0]]['id']=='mi_class' else cfg['emotion_per_class'] if d.specs[k[0]]['id']=='emotion_class' else 1 for k in d.groups}
    sampler=BalancedCoverageSampler(d.groups,counts,cfg['seed']);seen={q:[] for q in qs}
    for _ in range(args.step):
        rr=d.records[sampler.next_batch().reshape(-1)]
        for q in qs:seen[q].extend(lookup[int(i)] for i in rr[rr[:,1]==q,0])
    val=data[1];panel=val.records[np.asarray(cfg['validation_ids'])]
    protocol=dict(step_budget=args.step,config_sha256=digest(config_path),seed=42,alphas=[10,1000,100000],
        inner_subjects=inner_subjects,features='frozen REVE; frontal and posterior channel means, all four time patches retained, 4096 dimensions',
        selection='train-internal held-out subjects, dataset-macro balanced accuracy; fit scaler only within each fit partition',
        training_budget='exact replay of current attribute sample stream, including repeats; no extra attribute labels',
        limitation='Pooled linear probe cannot prove unpooled/nonlinear information absent. Repeated source validation is not blind. Encoder pretraining overlap not excluded.',
        code_sha256=digest(__file__),token_audit_sha256=digest('artifacts/jev_paired_qa/tokens/verified.json'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'label_audit.json').write_text(json.dumps(audit,indent=2))
    results={}
    for j,q in enumerate(qs):
        ii=np.array(seen[q]);fit=ii[~inner[ii]];valid=ii[inner[ii]];scores=[]
        assert len(fit)>0 and len(valid)>0 and train[ii].all()
        for a in protocol['alphas']:
            model=make_pipeline(StandardScaler(),Ridge(alpha=a,solver='cholesky'))
            model.fit(x[fit],np.eye(3)[y[fit,j]])
            pp=model.predict(x[valid]).argmax(1)
            scores.append(float(np.mean([balanced_accuracy_score(y[valid,j][dataset[valid]==ds],pp[dataset[valid]==ds]) for ds in sorted(set(dataset))])))
        alpha=protocol['alphas'][int(np.argmax(scores))]
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'));model.fit(x[ii],np.eye(3)[y[ii,j]])
        scores_all=model.predict(x[test]);pred=scores_all.argmax(1)
        panel_ids=np.array([lookup[int(i)] for i in panel[panel[:,1]==q,0]])
        panel_pred=model.predict(x[panel_ids]).argmax(1)
        result=dict(alpha=alpha,inner_scores=scores,training_draws=len(ii),unique_training_signals=len(set(ii)),
            training=metrics(y[ii,j],model.predict(x[ii]).argmax(1)),
            full_validation={ds:metrics(y[test,j][dataset[test]==ds],pred[dataset[test]==ds]) for ds in sorted(set(dataset))},
            exact_jev_panel={ds:metrics(y[panel_ids,j][dataset[panel_ids]==ds],panel_pred[dataset[panel_ids]==ds]) for ds in sorted(set(dataset))})
        results[specs[j]['id']]=result
        np.savez(out/f"{specs[j]['id']}_predictions.npz",source_ids=ids[test],labels=y[test,j],scores=scores_all,predictions=pred,training_source_ids=ids[ii],panel_source_ids=ids[panel_ids],panel_predictions=panel_pred)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(specs[j]['id'],{ds:round(m['balanced_accuracy'],4) for ds,m in result['full_validation'].items()},flush=True)
    (out/'complete.json').write_text(json.dumps(dict(results_sha256=digest(out/'results.json'),protocol_sha256=digest(out/'protocol.json'),label_audit_sha256=digest(out/'label_audit.json')),indent=2))


if __name__=='__main__':main()
