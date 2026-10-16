"""Audit candidate text representations before EEG-text contrastive training."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from eeg_align.model import load_text_only
from eeg_align.cache_multitask_v30 import digest

OUT=Path('artifacts/eeg_text_alignment/text_bank')


def diagnostic(vectors,rows):
    z=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    train=np.array([r['split']=='train' for r in rows]);y=np.array([r['class_id'] for r in rows])
    centers=np.stack([z[train&(y==c)].mean(0) for c in range(5)])
    centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    scores=z[~train]@centers.T;pred=scores.argmax(1);truth=y[~train]
    gram=z@z.T;off=~np.eye(len(z),dtype=bool);same=y[:,None]==y[None,:]
    return dict(heldout_wording_accuracy=float(np.mean(pred==truth)),heldout_labels=truth.tolist(),
        heldout_predictions=pred.tolist(),heldout_scores=scores.tolist(),class_prototype_cosine=(centers@centers.T).tolist(),
        mean_same_class_cosine=float(gram[same&off].mean()),mean_other_class_cosine=float(gram[~same].mean()),
        note='Text-only retrieval sanity check; no EEG, no evidence of unseen-task generalization.')


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    config=Path('configs/text_descriptions.json');rows=json.loads(config.read_text())['descriptions']
    assert len(rows)==30 and len({r['text'] for r in rows})==30
    for c in range(5):
        assert sum(r['class_id']==c and r['split']=='train' for r in rows)==4
        assert sum(r['class_id']==c and r['split']=='heldout_wording' for r in rows)==2
    base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());path=base['qwen_checkpoint']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    token=tok.encode('A',add_special_tokens=False);assert len(token)==1
    model,_,audit=load_text_only(path,token,'cuda:0',torch.bfloat16)
    model.eval().requires_grad_(False);arrays={'mean':[],'last':[]}
    for r in rows:
        ids=torch.tensor(tok.encode(r['text'],add_special_tokens=False),device='cuda')[None]
        z=model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state.float()[0]
        arrays['mean'].append(z.mean(0).cpu().numpy());arrays['last'].append(z[-1].cpu().numpy())
    reports={}
    for name,values in arrays.items():
        x=np.stack(values);assert np.isfinite(x).all()
        np.save(OUT/f'{name}.npy',x);reports[name]=diagnostic(x,rows)
    (OUT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (OUT/'complete.json').write_text(json.dumps(dict(checkpoint=path,loading=audit,
        checkpoint_index_sha256=digest(Path(path)/'model.safetensors.index.json'),
        descriptions_sha256=digest(config),reports=reports,
        hashes={name:digest(OUT/name) for name in ['mean.npy','last.npy','rows.jsonl']}),indent=2))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
