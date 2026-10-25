import math

from scripts.log_tensorboard import training_scalars, validation_scalars, jsonl


def test_validation_macro_pooled_and_semantic_loss_are_distinct():
    report={'normal':{
        'large':{'epoch':{'n':4,'accuracy':.75,'balanced_accuracy':.5}},
        'small':{'epoch':{'n':1,'accuracy':0.,'balanced_accuracy':0.}},
    }}
    rows=[{'dataset':'large','label':label,'probabilities':[.8,.2]} for label in [0,0,0,1]]
    rows.append({'dataset':'small','label':0,'probabilities':[.1,.9]})
    s=validation_scalars(report,rows)
    assert s['validation/overall/accuracy_macro']==.375
    assert s['validation/overall/accuracy_pooled']==.6
    assert s['validation/overall/balanced_accuracy_macro']==.25
    large_loss=-(3*math.log(.8)+math.log(.2))/4
    assert math.isclose(s['validation/overall/loss_macro'],(large_loss-math.log(.1))/2)
    assert math.isclose(s['validation/overall/loss_pooled'],(4*large_loss-math.log(.1))/5)


def test_train_preserves_overall_and_dataset_metrics_and_partial_tail(tmp_path):
    row={'step':7,'loss':.7,'train_accuracy':.6,
         'train_by_dataset':{'MI':{'loss':.8,'accuracy':.5},'SEED':{'loss':.6,'accuracy':.7}}}
    s=training_scalars(row)
    assert s['train/overall/accuracy']==.6 and s['train/overall/loss']==.7
    assert s['train/datasets/MI/accuracy']==.5 and s['train/datasets/SEED/loss']==.6
    path=tmp_path/'metrics.jsonl';path.write_text('{"step": 1}\n{"step":')
    assert jsonl(path)==[{'step':1}]
