"""Audit candidate text representations before EEG-text contrastive training."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from eeg_align.cache import digest

OUT=Path('artifacts/eeg_text_alignment/embedding_bank')


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
    if (OUT/'complete.json').exists():raise FileExistsError('Text bank is already complete')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    config=Path('configs/text_descriptions.json');rows=json.loads(config.read_text())['descriptions']
    assert len(rows)==30 and len({r['text'] for r in rows})==30
    for c in range(5):
        assert sum(r['class_id']==c and r['split']=='train' for r in rows)==4
        assert sum(r['class_id']==c and r['split']=='heldout_wording' for r in rows)==2
    path=Path('artifacts/models/Qwen3-Embedding-0.6B')
    downloaded=json.loads((path/'download_complete.json').read_text())
    for f in downloaded['files']:
        assert digest(path/f['file'])==f['sha256']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
    model,audit=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.float32,output_loading_info=True)
    assert not any(audit.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.cuda().eval().requires_grad_(False)
    arrays={'last':[]}
    for r in rows:
        batch=tok(r['text'],return_tensors='pt').to('cuda')
        z=model(**batch,use_cache=False,return_dict=True).last_hidden_state.float()[0,-1]
        arrays['last'].append(z.cpu().numpy())
    reports={}
    for name,values in arrays.items():
        x=np.stack(values);assert x.shape==(30,1024) and np.isfinite(x).all()
        np.save(OUT/f'{name}.npy',x);reports[name]=diagnostic(x,rows)
    (OUT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (OUT/'complete.json').write_text(json.dumps(dict(checkpoint=str(path),loading={k:sorted(v) if isinstance(v,set) else v for k,v in audit.items()},
        checkpoint_provenance=downloaded,
        descriptions_sha256=digest(config),reports=reports,
        hashes={name:digest(OUT/name) for name in ['last.npy','rows.jsonl']}),indent=2))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
