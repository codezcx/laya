"""Current alignment external metrics; target sets remain exploratory."""
import numpy as np

def metrics(rows,pred):
    y=np.array([r['class_id'] for r in rows]);pred=np.asarray(pred);people={}
    assert pred.shape==y.shape
    classes=np.unique(y)
    for s in sorted({r['subject'] for r in rows}):
        ii=np.array([i for i,r in enumerate(rows) if r['subject']==s]);yy=y[ii];pp=pred[ii]
        assert np.array_equal(np.unique(yy),classes)
        people[s]=dict(accuracy=float(np.mean(pp==yy)),balanced_accuracy=float(np.mean([np.mean(pp[yy==c]==c) for c in classes])))
    values=np.array([v['balanced_accuracy'] for v in people.values()])
    boot=np.random.default_rng(2026).choice(values,(10000,len(values)),replace=True).mean(1)
    return dict(n=len(rows),accuracy=float(np.mean(pred==y)),subject_accuracy=float(np.mean([v['accuracy'] for v in people.values()])),
        subject_balanced_accuracy=float(values.mean()),subject_balanced_95ci=np.quantile(boot,[.025,.975]).tolist(),per_subject=people)


def evaluate_run(folder,freeze):
    import json
    from pathlib import Path
    import torch
    from .cache import digest
    from .model import EEGTextAligner
    folder=Path(folder);record=json.loads(Path(freeze).read_text())
    assert record['runs'][folder.name]==digest(folder/'best.pt')
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);state=saved['model']
    model=EEGTextAligner(state['text_vectors'],state['text_labels'],saved['config']['control'])
    model.load_state_dict(state);model=model.cuda().eval();torch.set_num_threads(4)
    rows=[];scores=[];provenance={}
    for shard in range(4):
        root=Path('artifacts/multitask_bridge_v30/external')/str(shard)
        audit=json.loads((root/'complete.json').read_text())
        for name,sha in audit['hashes'].items():assert digest(root/name)==sha
        provenance[str(root)]=digest(root/'complete.json')
        rr=[json.loads(s) for s in (root/'rows.jsonl').read_text().splitlines()]
        xx=np.load(root/'tokens.npy',mmap_mode='r');lengths=np.load(root/'lengths.npy')
        assert len(rr)==len(xx)==len(lengths)
        with torch.no_grad():
            for i in range(0,len(xx),256):
                n=lengths[i:i+256];k=int(n.max())
                x=torch.tensor(np.array(xx[i:i+256,:k]),device='cuda',dtype=torch.float32)
                m=torch.arange(k,device='cuda')[None]<torch.tensor(n,device='cuda')[:,None]
                scores.append(model(x,m)['logits'].cpu().numpy())
        rows.extend(rr)
    logits=np.concatenate(scores);pred=logits.argmax(1);results={}
    for ds in sorted({r['dataset'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['dataset']==ds]
        results[ds]=metrics([rows[i] for i in ids],pred[ids])
    np.save(folder/'external_logits.npy',logits)
    output=dict(datasets=results,source_freeze_sha256=digest(freeze),cache_provenance=provenance,
        caveat='Exploratory: targets previously inspected; REVE pretraining includes 2b.')
    (folder/'external.json').write_text(json.dumps(output,indent=2));return output

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--freeze',required=True)
    args=p.parse_args();evaluate_run(args.run,args.freeze)
