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
