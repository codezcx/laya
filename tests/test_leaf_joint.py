import numpy as np
from collections import Counter
from eeg_align.train_joint import sample_indices,make_variant
from eeg_align.leaf_data import TASKS,metrics

def test_balanced_sampling_and_all_options():
 rows=[dict(task=t,label=y,subject=str(s),parent_id=f'{t}-{y}-{s}',id=f'{t}-{y}-{s}') for t in range(2) for y in range(len(TASKS[t]['classes'])) for s in range(2)]
 ix=sample_indices(rows,1);assert ix==sample_indices(rows,1)
 assert Counter(rows[i]['task'] for i in ix)=={0:24,1:24}
 for t in range(2):
  c=Counter(rows[i]['label'] for i in ix if rows[i]['task']==t);assert len(set(c.values()))==1
 for task in TASKS:
  for label in range(len(task['classes'])):
   for variant in [None,0,1]:
    q=make_variant(task,label,'sample',5,42,variant);assert q['option_class_ids'][q['label']]==task['classes'][label]['class_id']

def test_parent_aggregation_and_task_metrics():
 rows=[];p=[]
 for t in range(2):
  for label in range(len(TASKS[t]['classes'])):
   for w in range(label+1):
    rows.append(dict(task=t,label=label,parent_id=f'{t}-{label}'));p.append(np.eye(4)[label])
 m=metrics(rows,np.stack(p));assert m['2a']['trial']['n']==4 and m['SEED']['trial']['n']==3
 assert all(v['trial']['balanced_accuracy']==1 for v in m.values())
