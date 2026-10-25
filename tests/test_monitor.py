import json
from eeg_align.monitor import summarize_branch,update_monitor


def test_monitor_detects_collapse_without_calling_majority_accuracy_success(tmp_path):
    p=tmp_path/'updating';p.mkdir()
    m=dict(n=100,accuracy=.9,balanced_accuracy=.5,confusion_matrix=[[90,0],[10,0]])
    rows=[dict(step=101,loss=.7,train_accuracy=.5,gradient_norm=1.,train_by_dataset={'task':dict(n=8,loss=.7,accuracy=.5)})]
    rows.append(dict(step=200,validation=dict(normal={'task':dict(epoch=m)},consistency={'task':1.},selection_score=.5)))
    (p/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"partial":')
    result=summarize_branch(p)
    assert result['training_windows'][0]['accuracy']==.5
    assert result['validation_history'][0]['datasets']['task']['recall']==[1.,0.]
    assert result['alerts'][0]['kind']=='class_collapse'
    (tmp_path/'frozen').mkdir()
    update_monitor(tmp_path,['updating','frozen'])
    assert (tmp_path/'monitor.json').exists() and '90.00%' in (tmp_path/'LIVE.md').read_text()
