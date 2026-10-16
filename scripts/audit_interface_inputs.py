"""Source-only input consistency and linear fitting feasibility; no generalization claim."""
import json,hashlib
from pathlib import Path
import numpy as np
from eeg_align.cache import ROOT,digest
from scripts.report_interface_diagnostic import recompute

OUT=Path('artifacts/jev_interface_diagnostic')

def main():
    design=json.loads((OUT/'design.json').read_text())
    assert digest(ROOT/'rows.jsonl')==design['manifest_sha256']
    rows=[json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
    ids=design['indices'];assert len(ids)==len(set(ids))==60
    assert all(rows[i]['split']=='train' for i in ids)
    location={};arrays=[]
    for shard in range(4):
        folder=ROOT/str(shard);arrays.append(np.load(folder/'tokens.npy',mmap_mode='r'))
        for local,idx in enumerate(np.load(folder/'indices.npy')):location[int(idx)]=(shard,local)
    x=np.stack([arrays[location[i][0]][location[i][1]] for i in ids]).astype(np.float64).reshape(len(ids),-1)
    assert np.isfinite(x).all();labels=np.array([rows[i]['class_id'] for i in ids])
    seen={};conflicts=[]
    for i,vector in zip(ids,x):
        sha=hashlib.sha256(vector.tobytes()).hexdigest()
        if sha in seen and rows[seen[sha]]['class_id']!=rows[i]['class_id']:conflicts.append([seen[sha],i])
        seen[sha]=i
    # Centering and one global scalar preserve linear readout feasibility.
    centered=x-x.mean(0,keepdims=True);centered/=np.sqrt(np.mean(centered**2))
    gram=centered@centered.T/centered.shape[1]
    ridge=1e-6*np.trace(gram)/len(gram)
    target=np.eye(5)[labels];mean=target.mean(0)
    dual=np.linalg.solve(gram+ridge*np.eye(len(gram)),target-mean)
    prediction=gram@dual+mean
    result=dict(n=len(ids),unique_feature_vectors=len(seen),conflicting_duplicate_labels=conflicts,
        centered_sample_gram_rank=int(np.linalg.matrix_rank(gram)),max_centered_rank=len(ids)-1,
        linear_training_fit=recompute(prediction,labels),
        regularization=float(ridge),source_design_sha256=digest(OUT/'design.json'),
        note='Ridge fit uses the same 60 labeled training examples. Only a fitting-feasibility check, not a matched optimization baseline or zero-shot evidence.')
    # These least-squares scores are not calibrated logits; do not compare their CE to JEV training loss.
    result['linear_training_fit'].pop('loss')
    np.save(OUT/'linear_feasibility_logits.npy',prediction)
    (OUT/'input_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert not conflicts

if __name__=='__main__':main()
