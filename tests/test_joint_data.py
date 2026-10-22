import numpy as np
from collections import Counter
from eeg_align.joint_data import Sampler,TASKS
from eeg_align.train_joint import summarise

def test_task_class_parent_sampler():
    rows=[]
    for t,task in enumerate(TASKS):
        for label in range(len(task['classes'])):
            for parent in range(3):
                for window in range(parent+1):
                    rows.append(dict(task=t,label=label,dataset=f'd{t}',parent_id=f'{t}-{label}-{parent}'))
    s=Sampler(rows)
    assert s.batch(5)==s.batch(5) and s.batch(5)!=s.batch(6)
    for step in range(10):
        ix=s.batch(step);assert len(ix)==64
        assert Counter(rows[i]['task'] for i in ix)=={0:32,1:16,2:16}
        for rank in range(2):assert Counter(rows[i]['task'] for i in ix[rank::2])=={0:16,1:8,2:8}
        for task in range(3):
            counts=Counter(rows[i]['label'] for i in ix if rows[i]['task']==task)
            assert max(counts.values())-min(counts.values())<=1

def test_metrics_keep_task_and_trial_denominators():
    rows=[]
    for task in range(3):
        k=len(TASKS[task]['classes'])
        for label in range(k):
            for window in range(label+1):
                prob=np.eye(k)[label].tolist()
                rows.append(dict(id=f'{task}-{label}-{window}',parent_id=f'{task}-{label}',dataset=f'd{task}',task=task,label=label,probabilities=prob))
    m=summarise(rows);assert m['selection_score']==1.
    assert m['datasets']['d2']['window']['n']==28
    assert m['datasets']['d2']['trial']['n']==7
