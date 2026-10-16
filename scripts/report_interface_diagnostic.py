"""Independently audit tiny-set outputs; never turn fitting scores into zero-shot claims."""
import json
from pathlib import Path
import numpy as np
from eeg_align.cache import ROOT,digest

ROOT_OUT=Path('artifacts/jev_interface_diagnostic')

def recompute(logits,labels):
    assert logits.shape==(len(labels),5) and np.isfinite(logits).all()
    predicted=logits.argmax(1);shift=logits-logits.max(1,keepdims=True)
    loss=float(np.mean(np.log(np.exp(shift).sum(1))-shift[np.arange(len(labels)),labels]))
    return dict(MI=float(np.mean([np.mean(predicted[labels==c]==c) for c in [3,4]])),
        emotion=float(np.mean([np.mean(predicted[labels==c]==c) for c in [0,1,2]])),
        accuracy=float(np.mean(predicted==labels)),loss=loss,
        prediction_counts=np.bincount(predicted,minlength=5).tolist())

def check(actual,recorded):
    for key in ['MI','emotion','accuracy','loss']:
        assert np.isclose(actual[key],recorded[key],rtol=1e-5,atol=1e-6),(key,actual[key],recorded[key])
    assert actual['prediction_counts']==recorded['prediction_counts']

def main():
    design=json.loads((ROOT_OUT/'design.json').read_text())
    assert digest(ROOT/'rows.jsonl')==design['manifest_sha256']
    for name,sha in design['code_hashes'].items():assert digest(name)==sha
    all_rows=[json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
    rows=[all_rows[i] for i in design['indices']]
    assert len(rows)==len(set(design['indices']))==60 and all(r['split']=='train' for r in rows)
    groups={}
    for r in rows:groups[(r['dataset'],r['class_id'])]=groups.get((r['dataset'],r['class_id']),0)+1
    assert len(groups)==12 and set(groups.values())=={5}
    labels=np.array([r['class_id'] for r in rows]);report={}
    lines=['# JEV接口诊断：仅源训练小集','',
           '60个固定源训练窗口，不是验证/外部测试成绩。LoRA始终启用但冻结。', '',
           '| 条件 | 最后记录步数 | 拟合评估步数 | 小集MI | 小集情绪 | 同任务打乱MI | 同任务打乱情绪 | 任务平均MI | 任务平均情绪 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in design['variants']:
        folder=ROOT_OUT/name;record={};logs=[]
        if (folder/'metrics.jsonl').exists():
            text=(folder/'metrics.jsonl').read_text()
            for line in text.splitlines():
                try:logs.append(json.loads(line))
                except json.JSONDecodeError:
                    if line!=text.splitlines()[-1]:raise
        fits=[r for r in logs if 'fit' in r]
        record['last_step']=max((r['step'] for r in logs),default=0)
        record['last_fit']=fits[-1] if fits else None
        if (folder/'config.json').exists():record['config']=json.loads((folder/'config.json').read_text())
        if (folder/'complete.json').exists():
            final=json.loads((folder/'complete.json').read_text())
            assert final['final_fit']['step']==design['steps']
            checks={}
            for file,key,value in [('fit_logits.npy','fit',final['final_fit']['fit']),
                    ('shuffled_logits.npy','shuffled',final['shuffled_within_task']),('task_mean_logits.npy','task_mean',final['task_mean'])]:
                result=recompute(np.load(folder/file),labels);check(result,value);checks[key]=result
            gate=all(checks['fit'][t]>=.95 for t in ['MI','emotion'])
            assert gate==final['fit_gate_passed']
            record.update(completed=True,independent_scores=checks,fit_gate_passed=gate,
                shuffle_drop={t:checks['fit'][t]-checks['shuffled'][t] for t in ['MI','emotion']},
                artifact_hashes={f:digest(folder/f) for f in ['fit_logits.npy','shuffled_logits.npy','task_mean_logits.npy','connector.pt','complete.json']})
            values=[checks[k][t] for k in ['fit','shuffled','task_mean'] for t in ['MI','emotion']]
            lines.append(f'| {name} | 200 | 200 | '+' | '.join(f'{v*100:.2f}%' for v in values)+' |')
        else:
            record['completed']=False
            f=record['last_fit'];values='pending | pending' if f is None else f"{f['fit']['MI']*100:.2f}% | {f['fit']['emotion']*100:.2f}%"
            lines.append(f"| {name} | {record['last_step']} | {f['step'] if f else 'pending'} | {values} | pending | pending | pending | pending |")
        report[name]=record
    lines+=['','95%仅为训练小集拟合门槛，不是zero-shot成功标准。未完成组不做排名或提前选方案。']
    (ROOT_OUT/'independent_report.json').write_text(json.dumps(report,indent=2))
    (ROOT_OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
