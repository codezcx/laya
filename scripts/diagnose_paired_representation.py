"""CPU-only matched probes: locate losses of MI/class or band-change information."""
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix
from eeg_align.paired_cache import PairedTokenCache
from eeg_align.cache import digest
from eeg_tasks.attribute_questions import absolute_power,mi_specifications

OUT=Path('artifacts/jev_paired_qa/diagnostics')


def main():
    OUT.mkdir(exist_ok=False)
    specs=mi_specifications();records=[];signals=[];targets=[]
    for split,root in [('train',Path('datasets/eeg_attribute_qa')),('validation',Path('datasets/eeg_attribute_qa_validation'))]:
        src=[json.loads(s) for s in (root/'mi_sources.jsonl').read_text().splitlines()]
        pair=np.stack([np.load(root/f'signals/mi_{seg}.npy') for seg in ['baseline','task']],axis=1)
        signals.append(pair)
        yy=[]
        for spec in specs:
            rows=[json.loads(s) for s in (root/'questions'/f'{spec["id"]}.jsonl').read_text().splitlines()]
            assert all(r['input']['segments'][0]['signal_index']==i for i,r in enumerate(rows))
            yy.append(['ABC'.index(r['target']['answer']) for r in rows])
        targets.append(np.array(yy).T)
        records.extend(dict(source_id=r['source_row_index'],subject=r['source_metadata']['subject'],split=split) for r in src)
    pair=np.concatenate(signals);y=np.concatenate(targets)
    train=np.array([r['split']=='train' for r in records]);subject=np.array([r['subject'] for r in records])
    people=np.unique(subject[train]);np.random.default_rng(42).shuffle(people)
    inner=np.isin(subject,people[:9]) & train;fit=train & ~inner;test=~train
    assert not set(subject[fit]) & set(subject[inner]) and not set(subject[train]) & set(subject[test])
    widths=[len(s['options']) for s in specs];starts=np.cumsum([0]+widths)
    target=np.concatenate([np.eye(k)[y[:,j]] for j,k in enumerate(widths)],axis=1)
    power=absolute_power(pair);logpower=np.concatenate([10*np.log10(power[b]) for b in ['alpha','beta']],axis=-1)
    # [N,segment,channel*band]; ratios preserve the shared amplitude convention.
    delta=logpower[:,1]-logpower[:,0]
    features={'band_power':np.concatenate([logpower.reshape(len(y),-1),delta],axis=1)}
    cache=PairedTokenCache();zs=[]
    for begin in range(0,len(records),128):
        z,mask,cuts=cache.batch([r['source_id'] for r in records[begin:begin+128]],'cpu')
        assert all(c==9 for c in cuts) and mask.all()
        zs.append(z.numpy())
    tokens=np.concatenate(zs).reshape(len(y),2,3,3,512)
    features['reve_channel_pooled']=tokens.mean(axis=3).reshape(len(y),-1)
    features['reve_task_only']=tokens[:,1].mean(axis=2).reshape(len(y),-1)
    protocol=dict(train_subjects=sorted(set(subject[fit])),inner_validation_subjects=sorted(set(subject[inner])),
        source_validation_subjects=sorted(set(subject[test])),n=dict(fit=int(fit.sum()),inner=int(inner.sum()),source_validation=int(test.sum())),
        selection='alpha selected by inner subject-held-out balanced accuracy for each question, then refit all source training subjects',
        alphas=[10,1000,100000],limitations='Pooled REVE probe cannot establish information absence in unpooled/nonlinear features; repeated source validation is not blind.',
        code_sha256=digest(__file__),token_audit_sha256=digest('artifacts/jev_paired_qa/tokens/verified.json'))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results={}
    for name,x in features.items():
        x=x.astype(np.float64);scores=[]
        for alpha in protocol['alphas']:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
            model.fit(x[fit],target[fit]);pred=model.predict(x[inner])
            scores.append([balanced_accuracy_score(y[inner,j],pred[:,starts[j]:starts[j+1]].argmax(1)) for j in range(len(specs))])
            print(name,'inner',alpha,scores[-1],flush=True)
        chosen=np.argmax(scores,axis=0);result={}
        for k in set(chosen.tolist()):
            alpha=protocol['alphas'][k];model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
            model.fit(x[train],target[train]);pred=model.predict(x[test]);pred_train=model.predict(x[train])
            for j in np.where(chosen==k)[0]:
                pp=pred[:,starts[j]:starts[j+1]].argmax(1);pt=pred_train[:,starts[j]:starts[j+1]].argmax(1)
                result[specs[j]['id']]=dict(alpha=alpha,inner_balanced_accuracy=scores[k][j],
                    training_balanced_accuracy=balanced_accuracy_score(y[train,j],pt),
                    validation_accuracy=accuracy_score(y[test,j],pp),validation_balanced_accuracy=balanced_accuracy_score(y[test,j],pp),
                    confusion=confusion_matrix(y[test,j],pp,labels=np.arange(widths[j])).tolist(),
                    per_subject={s:balanced_accuracy_score(y[test,j][subject[test]==s],pp[subject[test]==s]) for s in sorted(set(subject[test]))})
                np.save(OUT/f'{name}_{specs[j]["id"]}_predictions.npy',pp)
        results[name]=result
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print('RESULT',name,json.dumps(result),flush=True)
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',results_sha256=digest(OUT/'results.json')),indent=2))


if __name__=='__main__':main()
