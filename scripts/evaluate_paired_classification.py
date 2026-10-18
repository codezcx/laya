"""Read-only checkpoint evaluation: per-subject MI, option swap, within-subject EEG shuffle."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eeg_align.attribute_data import AttributeData,QuestionTokens,semantic_predictions,draw_permutations
from eeg_align.paired_cache import PairedTokenCache
from eeg_align.jev import load_jev,JEVBridge
from eeg_align.cache import digest


def report(labels,predictions,subjects):
    labels=np.asarray(labels);predictions=np.asarray(predictions);subjects=np.asarray(subjects)
    def metric(ii):
        y,p=labels[ii],predictions[ii];cm=np.zeros((2,2),dtype=int)
        np.add.at(cm,(y,p),1);counts=cm.sum(1)
        return dict(n=len(y),accuracy=float(np.mean(y==p)),balanced_accuracy=float(np.mean(cm.diagonal()/counts)),confusion=cm.tolist())
    return dict(overall=metric(np.arange(len(labels))),subjects={str(s):metric(subjects==s) for s in sorted(set(subjects))})


def shuffled_ids(signal_ids,subjects,seed=1701):
    result=np.array(signal_ids,copy=True);rng=np.random.default_rng(seed);subjects=np.asarray(subjects)
    for s in sorted(set(subjects)):
        ii=np.where(subjects==s)[0];result[ii]=rng.permutation(result[ii])
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--memory-fraction',type=float,default=.48)
    args=p.parse_args()
    if args.batch_size<1:raise ValueError('Positive batch size required')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,torch.device(args.device))
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint_sha=digest(args.checkpoint);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    data=AttributeData('datasets/eeg_attribute_qa_validation','validation')
    assert saved['config']['validation_data_sha256']==data.root_hash
    assert saved['config']['token_audit_sha256']==digest('artifacts/jev_paired_qa/tokens/verified.json')
    q=next(i for i,s in enumerate(data.specs) if s['id']=='mi_class');ids=np.where(data.records[:,1]==q)[0];records=data.records[ids]
    source=[json.loads(s) for s in Path('datasets/eeg_attribute_qa_validation/mi_sources.jsonl').read_text().splitlines()]
    people={r['source_row_index']:r['source_metadata']['subject'] for r in source}
    subjects=np.array([people[int(i)] for i in records[:,0]]);labels=records[:,2]
    assert len(records)==1000 and set(labels)=={0,1}
    cache=PairedTokenCache();text,w,b,tok=load_jev(args.device)
    model=JEVBridge(text,w,b,tok,train_lora=False)
    assert set(saved['trainable'])==set(saved['config']['trainable_names'])
    assert set(saved['trainable'])<=set(dict(model.named_parameters()))
    incompatible=model.load_state_dict(saved['trainable'],strict=False);assert not incompatible.unexpected_keys
    model.requires_grad_(False).eval();questions=QuestionTokens(tok,data.specs,device=args.device)
    step=saved['step'];del saved
    random_order=draw_permutations(data,data.records,np.random.default_rng(20261018))[ids]
    signal=records[:,0];shuffled=shuffled_ids(signal,subjects)
    cases={'matched':(signal,random_order),'swapped_options':(signal,1-random_order),'shuffled_within_subject':(shuffled,random_order)}
    results={};all_predictions={}
    for name,(signal_ids,order) in cases.items():
        predictions=[];scores=[]
        for start in range(0,len(ids),args.batch_size):
            if start % (args.batch_size*10)==0:print(name,start,len(ids),flush=True)
            sl=slice(start,start+args.batch_size);z,mask,cuts=cache.batch(signal_ids[sl],args.device)
            suffix,_=questions.batch(records[sl],order[sl])
            logits=model(z,mask,suffix_ids=suffix,num_choices=2,segment_splits=cuts)['logits']
            letters=logits.argmax(1).cpu().numpy()
            predictions.extend(semantic_predictions(letters,order[sl],[2]*len(letters)).tolist())
            scores.extend(logits.float().cpu().tolist())
        predictions=np.array(predictions);all_predictions[name]=predictions
        results[name]=report(labels,predictions,subjects)
        np.savez(args.output/f'{name}.npz',source_ids=signal_ids,subjects=subjects,labels=labels,predictions=predictions,option_permutations=order,letter_logits=np.array(scores))
        (args.output/'results.json').write_text(json.dumps(results,indent=2));print(name,results[name],flush=True)
    results['option_swap_agreement']=float(np.mean(all_predictions['matched']==all_predictions['swapped_options']))
    results['matched_minus_shuffled']=results['matched']['overall']['accuracy']-results['shuffled_within_subject']['overall']['accuracy']
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    assert checkpoint_sha==digest(args.checkpoint),'Checkpoint changed during evaluation; use immutable copy'
    (args.output/'complete.json').write_text(json.dumps(dict(step=step,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_sha,
        data_sha256=data.root_hash,code_sha256=digest(__file__),results_sha256=digest(args.output/'results.json')),indent=2))


if __name__=='__main__':main()
