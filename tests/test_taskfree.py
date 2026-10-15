import torch
from eeg_tasks.taskfree_train import joint_metrics,split_indices

def test_joint_scores_do_not_mask_wrong_task():
 rows=[dict(dataset='emotion',subject='1',class_id=0,task=0),dict(dataset='mi',subject='1',class_id=3,task=1)]
 scores=joint_metrics(rows,torch.tensor([[2.,0.,0.,4.,0.],[4.,0.,0.,2.,0.]]))
 assert scores['macro_task']['joint_balanced']==0
 assert scores['macro_task']['task_accuracy']==0
 assert scores['macro_task']['oracle_balanced']==1

def test_dataset_holdout_excludes_every_subject_and_split():
 rows=[dict(dataset=d,subject=s,split=sp) for d in ['SEED','FACED-NEMAR','Cho2017','Dreyer2023','Lee2019'] for s,sp in [('1','train'),('2','validation')]]
 tr,va,te=split_indices(rows,'FACED-NEMAR')
 assert len(tr)==len(va)==3 and len(te)==4
 assert {rows[i]['dataset'] for i in tr+va}=={'SEED','Cho2017','Dreyer2023'}
