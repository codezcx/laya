"""Evaluate a fixed JEV checkpoint on every source training trial, dataset by dataset."""
import argparse,json,os
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from eeg_align.jev import load_jev,JEVBridge
from eeg_align.mi_training_data import Tokens,digest,write_json,metrics
from eeg_align.train_jev import evaluate

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(4)
dist.init_process_group('nccl',timeout=timedelta(minutes=60))
if rank==0:a.out.mkdir(parents=True,exist_ok=False)
dist.barrier()
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert digest('datasets/mi_tokens/complete.json')==saved['config']['data_sha256']
for path in ['eeg_align/jev.py','eeg_align/mi_training_data.py','eeg_align/distributed.py','configs/encoder.json','configs/mi_train.json']:
 assert digest(path)==saved['config']['code_sha256'][path],path
text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,train_lora=True,train_head=True)
assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
model.load_state_dict(saved['trainable'],strict=False);step=saved['step'];del saved
model.eval().requires_grad_(False);data=Tokens('train');all_rows=data.rows;predictions=[]
priority=['BNCI2014_001','PhysionetMI','Weibo2014','Zhou2016']
order=priority+sorted({r['dataset'] for r in all_rows}-set(priority))
if rank==0:write_json(a.out/'config.json',dict(checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),step=step,split='train',trials=len(all_rows),dataset_order=order,script_sha256=digest(__file__),note='Fixed-checkpoint evaluation; no optimization, no validation/test data used as training examples.'))
for ds in order:
 data.rows=[r for r in all_rows if r['dataset']==ds]
 result,pred=evaluate(model,data,rank,world,microbatch=2)
 predictions+=pred
 if rank==0:
  write_json(a.out/f'{ds}.json',dict(step=step,split='train',metrics=result,predictions=pred))
  write_json(a.out/'progress.json',dict(last_dataset=ds,completed_trials=len(predictions),total_trials=len(all_rows)))
  print('TRAIN_EVAL',ds,len(pred),result['accuracy'],result['left_recall'],result['right_recall'],flush=True)
if rank==0:
 assert len(predictions)==len(all_rows) and len({r['id'] for r in predictions})==len(all_rows)
 result=metrics(predictions);result['loss']=sum(r['loss'] for r in predictions)/len(predictions)
 write_json(a.out/'metrics.json',dict(step=step,split='train',metrics=result))
 write_json(a.out/'complete.json',dict(step=step,trials=len(predictions),datasets=order))
dist.barrier();dist.destroy_process_group()
