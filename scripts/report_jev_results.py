"""Independently audit saved five-class predictions and frozen checkpoint identities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def recompute(rows,logits):
    assert logits.shape==(len(rows),5) and np.isfinite(logits).all()
    pred=logits.argmax(axis=1);result={}
    for dataset in sorted({r['dataset'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['dataset']==dataset]
        confusion=np.zeros((5,5),dtype=int);subjects={}
        for i in indices:confusion[rows[i]['class_id'],pred[i]]+=1
        for subject in sorted({rows[i]['subject'] for i in indices}):
            ids=[i for i in indices if rows[i]['subject']==subject]
            cm=np.zeros((5,5),dtype=int)
            for i in ids:cm[rows[i]['class_id'],pred[i]]+=1
            counts=cm.sum(axis=1);valid=counts>0
            subjects[subject]=dict(accuracy=float(np.trace(cm)/cm.sum()),
                balanced_accuracy=float((cm.diagonal()[valid]/counts[valid]).mean()))
        result[dataset]=dict(n=len(indices),accuracy=float(np.trace(confusion)/confusion.sum()),
            subject_accuracy=float(np.mean([s['accuracy'] for s in subjects.values()])),
            subject_balanced_accuracy=float(np.mean([s['balanced_accuracy'] for s in subjects.values()])),
            per_subject=subjects,confusion_matrix=confusion.tolist(),prediction_counts=confusion.sum(axis=0).tolist())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=Path('artifacts/jev_eeg_coverage/runs'))
    args=p.parse_args();freeze_path=args.runs/'source_freeze.json'
    freeze=json.loads(freeze_path.read_text());rows=[];cache_hashes={}
    for shard in range(4):
        folder=Path('artifacts/multitask_bridge_v30/external')/str(shard)
        audit=json.loads((folder/'complete.json').read_text())
        assert audit['hashes']['rows.jsonl']==sha(folder/'rows.jsonl')
        cache_hashes[str(folder)]=sha(folder/'complete.json')
        rows.extend(json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines())
    results={}
    for name,expected_sha in freeze['runs'].items():
        root=args.runs/name;assert sha(root/'best.pt')==expected_sha
        saved=json.loads((root/'external.json').read_text())
        assert saved['source_freeze_sha256']==sha(freeze_path)
        assert saved['cache_provenance']==cache_hashes
        actual=recompute(rows,np.load(root/'external_logits.npy'))
        assert actual.keys()==saved['datasets'].keys()
        for ds,item in actual.items():
            old=saved['datasets'][ds];assert old['n']==item['n']
            for metric in ['accuracy','subject_accuracy','subject_balanced_accuracy']:
                assert abs(item[metric]-old[metric])<1e-12,(name,ds,metric)
            assert item['per_subject'].keys()==old['per_subject'].keys()
            for subject,v in item['per_subject'].items():
                for metric,value in v.items():assert abs(value-old['per_subject'][subject][metric])<1e-12
        results[name]=dict(checkpoint_sha256=expected_sha,logits_sha256=sha(root/'external_logits.npy'),datasets=actual)
    output=dict(verified=True,runs=results,source_freeze_sha256=sha(freeze_path),
        note='Exploratory targets already inspected; REVE pretraining includes 2b. No claim of untouched zero-shot evaluation.')
    (args.runs/'independent_results.json').write_text(json.dumps(output,indent=2))
    for name,r in results.items():
        print(name,[(ds,round(100*v['accuracy'],3),round(100*v['subject_balanced_accuracy'],3)) for ds,v in r['datasets'].items()])

if __name__=='__main__':main()
