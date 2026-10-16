"""Compare completed v31 source-prefix records with v30, excluding timings."""
import json
from pathlib import Path

ROOT = Path('artifacts/multitask_bridge_v31')
OLD = Path('artifacts/multitask_bridge_v30')


def lines(path):
    if not path.exists(): return []
    # A live writer may have an unfinished final line; never treat it as final.
    data=path.read_text()
    return [json.loads(s) for s in data.splitlines(keepends=True) if s.endswith('\n') and s.strip()]


def main():
    results={}
    for current,previous in [('qwen','qwen_frozen'),('signal','signal_control')]:
        new=ROOT/current;old=OLD/previous
        if not (new/'run.json').exists():
            results[current]={'status':'not_started'};continue
        a=json.loads((new/'run.json').read_text());b=json.loads((old/'run.json').read_text())
        # Budget is the sole allowed run-metadata difference.
        keys=set(a)|set(b)
        differences={k:[b.get(k),a.get(k)] for k in keys if a.get(k)!=b.get(k)}
        assert differences=={'steps':[500,3000]},differences
        oldval={r['step']:r for r in lines(old/'validation.jsonl')}
        newval={r['step']:r for r in lines(new/'validation.jsonl') if r['step']<=500}
        delta=[]
        for step,r in newval.items():
            ref=oldval[step];assert r['n']==ref['n']
            assert r['datasets'].keys()==ref['datasets'].keys()
            values=[abs(r['selection_score']-ref['selection_score'])]
            for ds in r['datasets']:
                values.extend(abs(r['datasets'][ds][k]-ref['datasets'][ds][k]) for k in ['joint','oracle','routing'])
            delta.append({'step':step,'maximum_metric_difference':max(values)})
        oldloss={r['step']:r['loss'] for r in lines(old/'train.jsonl')}
        losses=[{'step':r['step'],'absolute_loss_difference':abs(r['loss']-oldloss[r['step']])}
                for r in lines(new/'train.jsonl') if r['step']<=500]
        complete=set(newval)==set(oldval) and {r['step'] for r in losses}==set(oldloss)
        results[current]=dict(status='prefix_complete' if complete else 'prefix_running',
            metadata_difference=differences,validation=delta,losses=losses,
            max_loss_difference=max((r['absolute_loss_difference'] for r in losses),default=None),
            max_metric_difference=max((r['maximum_metric_difference'] for r in delta),default=None))
    path=ROOT/'prefix_audit.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(results,indent=2));tmp.replace(path)
    print(json.dumps({k:{f:v.get(f) for f in ['status','max_loss_difference','max_metric_difference']} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
