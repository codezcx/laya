import numpy as np
from scripts.report_jev_results import recompute


def test_subject_balancing_and_cross_task_error():
    rows=[dict(dataset='mi',subject='a',class_id=y) for y in [3,3,3,4]]
    rows += [dict(dataset='mi',subject='b',class_id=y) for y in [3,4]]
    logits=np.eye(5)[[3,3,0,3,3,4]]
    result=recompute(rows,logits)['mi']
    assert result['accuracy']==4/6
    assert abs(result['subject_balanced_accuracy']-2/3)<1e-12
    assert result['subject_accuracy']==.75
    assert result['confusion_matrix'][3][0]==1
    assert sum(result['prediction_counts'])==6


def test_reject_nonfinite_logits():
    rows=[dict(dataset='mi',subject='a',class_id=3)]
    try:recompute(rows,np.full((1,5),np.nan))
    except AssertionError:pass
    else:raise AssertionError('NaN predictions accepted')
