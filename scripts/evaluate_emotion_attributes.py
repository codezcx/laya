"""Immutable-snapshot emotion panel: probabilities and matched/shuffled/zero/option controls."""
import json
from pathlib import Path
import numpy as np
import torch
from eeg_align.cache import digest
from eeg_align.attribute_data import AttributeData,QuestionTokens,draw_permutations,semantic_predictions,summarize
from eeg_align.paired_cache import PairedTokenCache
from eeg_align.jev import load_jev,JEVBridge

OUT=Path('artifacts/jev_paired_qa/emotion_decision_audit')


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.48)
    snapshot=OUT/'snapshot.pt';sha=digest(snapshot)
    saved=torch.load(snapshot,map_location='cpu',weights_only=False);cfg=saved['config'];step=saved['step']
    data=AttributeData('datasets/eeg_attribute_qa_validation','validation',True)
    assert cfg['validation_classification_sha256']==data.classification_hash and cfg['validation_data_sha256']==data.root_hash
    ids=np.array([i for i in cfg['validation_ids'] if data.specs[data.records[i,1]]['task']=='emotion'])
    records=data.records[ids];signal=records[:,0].copy()
    orders=draw_permutations(data,data.records,np.random.default_rng(20261018),True,True)[ids]
    shuffled=signal.copy();rng=np.random.default_rng(1701);strata={}
    for j,(i,q,_,ds) in enumerate(records):strata.setdefault((int(q),int(ds),data.subjects[int(i)][1]),[]).append(j)
    for ii in strata.values():shuffled[ii]=rng.permutation(signal[ii])
    cache=PairedTokenCache();text,w,b,tok=load_jev('cuda:0');model=JEVBridge(text,w,b,tok)
    assert set(saved['trainable'])==set(cfg['trainable_names'])
    loaded=model.load_state_dict(saved['trainable'],strict=False);assert not loaded.unexpected_keys
    model.requires_grad_(False).eval();del saved
    questions=QuestionTokens(tok,data.specs);results={};all_preds={}
    cases={'matched':(signal,orders),'shuffled_within_subject':(shuffled,orders),
           'zero_signal':(signal,orders),'changed_option_order':(signal,(orders+1)%6)}
    for name,(signals,perms) in cases.items():
        probabilities=[];semantic_probabilities=[];pred=[];letters=[]
        for start in range(0,len(ids),6):
            rr=records[start:start+6];z,mask,cuts=cache.batch(signals[start:start+6],'cuda')
            if name=='zero_signal':z.zero_()
            suffix,labels=questions.batch(rr,perms[start:start+6])
            logits=model(z,mask,suffix_ids=suffix,num_choices=3,segment_splits=cuts)['logits']
            pp=torch.softmax(logits.float(),-1).cpu().numpy();ll=pp.argmax(1)
            probabilities.extend(pp);letters.extend(ll)
            pred.extend(semantic_predictions(ll,perms[start:start+6]).tolist())
            from eeg_align.attribute_data import PERMUTATIONS
            for prob,perm in zip(pp,perms[start:start+6]):
                semantic=np.zeros(3);semantic[list(PERMUTATIONS[int(perm)])]=prob;semantic_probabilities.append(semantic)
        pred=np.array(pred);probabilities=np.asarray(probabilities);sem=np.asarray(semantic_probabilities)
        report=summarize(data,ids,pred)
        report.pop('selection_score',None)
        report['signal_identity_changed_fraction']=float(np.mean(signals!=signal))
        for q,spec in enumerate(data.specs):
            ii=records[:,1]==q
            if not ii.any():continue
            report['questions'][spec['id']]['letter_counts']=np.bincount(np.asarray(letters)[ii],minlength=3).tolist()
            report['questions'][spec['id']]['mean_true_answer_probability']=float(sem[ii,records[ii,2]].mean())
            report['questions'][spec['id']]['mean_max_probability']=float(probabilities[ii].max(1).mean())
        results[name]=report;all_preds[name]=pred
        np.savez(OUT/f'{name}.npz',qa_indices=ids,source_ids=signals,labels=records[:,2],question_ids=records[:,1],dataset_ids=records[:,3],
                 permutations=perms,predictions=pred,letter_probabilities=probabilities,semantic_probabilities=sem)
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print(name,{q:round(z['accuracy'],4) for q,z in report['questions'].items()},flush=True)
    results['agreement_with_matched']={name:float(np.mean(pred==all_preds['matched'])) for name,pred in all_preds.items()}
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    assert digest(snapshot)==sha
    (OUT/'complete.json').write_text(json.dumps(dict(step=step,snapshot_sha256=sha,results_sha256=digest(OUT/'results.json'),code_sha256=digest(__file__),
        note='Softmax scores are uncalibrated. Some within-subject/question strata are small, including singleton strata. Mixed EEG lengths/batch padding can cause small numeric differences from distributed validation.',
        files={f.name:digest(f) for f in OUT.glob('*.npz')}),indent=2))


if __name__=='__main__':main()
