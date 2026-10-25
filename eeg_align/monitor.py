"""Read-only, reproducible summaries and anomaly flags for the current BN comparison."""
import json
from pathlib import Path
import numpy as np


def read_jsonl(path):
    if not path.exists():return []
    out=[]
    lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try:out.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    return out


def summarize_branch(folder):
    events=read_jsonl(folder/'metrics.jsonl');train=[r for r in events if 'loss' in r]
    validations=[r for r in events if 'validation' in r]
    latest=train[-1] if train else {};windows=[];alerts=[]
    for end in sorted({((r['step']-1)//100+1)*100 for r in train}):
        rr=[r for r in train if end-100<r['step']<=end]
        windows.append(dict(start_step=rr[0]['step'],end_step=rr[-1]['step'],updates=len(rr),
                            loss=float(np.mean([r['loss'] for r in rr])),
                            accuracy=float(np.mean([r['train_accuracy'] for r in rr])),
                            by_dataset={ds:dict(loss=float(np.mean([r['train_by_dataset'][ds]['loss'] for r in rr])),
                                                accuracy=float(np.mean([r['train_by_dataset'][ds]['accuracy'] for r in rr])))
                                        for ds in rr[0]['train_by_dataset']}))
    history=[]
    for row in validations:
        table={}
        for ds,v in row['validation']['normal'].items():
            m=v['epoch'];cm=np.asarray(m['confusion_matrix']);recall=np.diag(cm)/np.maximum(cm.sum(1),1)
            frac=cm.sum(0)/max(cm.sum(),1)
            table[ds]=dict(accuracy=m['accuracy'],balanced_accuracy=m['balanced_accuracy'],
                           recall=recall.tolist(),predicted_fraction=frac.tolist(),n=m['n'],
                           consistency=row['validation']['consistency'][ds])
            if np.min(recall)==0 or np.max(frac)>.95:
                alerts.append(dict(step=row['step'],dataset=ds,kind='class_collapse',recall=recall.tolist(),predicted_fraction=frac.tolist()))
            if history:
                old=history[-1]['datasets'][ds]['balanced_accuracy']
                if old-m['balanced_accuracy']>=.1:
                    alerts.append(dict(step=row['step'],dataset=ds,kind='BA_drop_at_least_10pp',previous=old,current=m['balanced_accuracy']))
        history.append(dict(step=row['step'],macro_ba=row['validation']['selection_score'],datasets=table,
                            checkpoint_reload=row.get('checkpoint_check')))
    if latest and any(not np.isfinite(latest[k]) for k in ['loss','gradient_norm']):
        alerts.append(dict(step=latest['step'],kind='nonfinite_training'))
    return dict(last_step=latest.get('step'),bn_state=latest.get('bn_state'),learning_rates=latest.get('learning_rates'),
                training_windows=windows,validation_history=history,alerts=alerts)


def update_monitor(root,policies):
    root=Path(root);report={policy:summarize_branch(root/policy) for policy in policies}
    checks={};files=[root/p/'initial_state.json' for p in policies]
    if all(p.exists() for p in files):
        proofs=[json.loads(p.read_text()) for p in files]
        for key in ['step','trainable_sha256','buffers_sha256','source_sha256','branch_seed']:
            checks[key]=all(p[key]==proofs[0][key] for p in proofs)
        if not all(checks.values()):raise RuntimeError(f'Branch initial states differ: {checks}')
    result=dict(initial_state_match=checks,branches=report)
    tmp=root/'monitor.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(root/'monitor.json')
    lines=['# Current BN comparison','',
           'Updating vs frozen BN; same initial checkpoint, optimizer, LR schedule, data and batch. '
           'Training numbers are sampled batch averages. Validation uses the fixed held-out subjects. '
           'Alerts are diagnostic flags, not proof of their cause.','',
           '| Branch | Last update | Latest 100-update block accuracy | Loss | Latest validation macro BA |',
           '| --- | ---: | ---: | ---: | ---: |']
    for policy,d in report.items():
        w=d['training_windows'][-1] if d['training_windows'] else None
        v=d['validation_history'][-1] if d['validation_history'] else None
        lines.append(f"| {policy} | {d['last_step']} | {w['accuracy']*100:.2f}% | {w['loss']:.4f} | "+(f"{v['macro_ba']*100:.2f}%" if v else 'pending')+' |' if w else f'| {policy} | pending | — | — | pending |')
    for policy,d in report.items():
        if not d['validation_history']:continue
        last=d['validation_history'][-1]
        lines+=['',f"## {policy}: validation update {last['step']}",'',
                '| Dataset | Accuracy | BA | Class recalls | Option consistency |','| --- | ---: | ---: | --- | ---: |']
        for ds,m in last['datasets'].items():
            lines.append(f"| {ds} | {m['accuracy']*100:.2f}% | {m['balanced_accuracy']*100:.2f}% | "+', '.join(f'{r*100:.1f}%' for r in m['recall'])+f" | {m['consistency']*100:.2f}% |")
        lines+=['',f"Cumulative diagnostic flags: {len(d['alerts'])}. Full history and flags: `monitor.json`."]
    tmp=root/'LIVE.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(root/'LIVE.md')
    return result
