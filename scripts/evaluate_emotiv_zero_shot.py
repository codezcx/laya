"""Fixed source-selected MI JEV checkpoint on external Emotiv; no target fitting."""
import argparse,json,os,math,time
from pathlib import Path
from datetime import timedelta
import numpy as np
from scipy.signal import resample_poly
import torch
import torch.distributed as dist
from transformers import AutoModel
from eeg_align.jev import load_jev,JEVBridge
from eeg_align.mi_training_data import digest,write_json,metrics,canonical_channel

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/zhangchenfei/prepared_left_right'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(3);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
if rank==0:a.out.mkdir(parents=True,exist_ok=False)
dist.barrier();began=time.monotonic();checkpoint=Path('artifacts/mi_jev_tuab_lr_continued/best.pt')
saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==2307
for f in ['eeg_align/jev.py','eeg_align/mi_training_data.py','configs/encoder.json']:
 assert digest(f)==saved['config']['code_sha256'][f],f
rows=[];arrays=[];files={};channels=None
for split in ['train','validation','test']:
 f=a.data/f'{split}.npz';files[str(f)]=digest(f)
 with np.load(f,allow_pickle=False) as d:
  x=d['X'];y=d['y'];names=d['channels'].tolist();assert int(d['sample_rate_hz'])==128 and x.shape[1:]==(14,636)
  if channels is None:channels=names
  assert channels==names and np.isfinite(x).all() and set(y.tolist())=={0,1}
  for i in range(len(x)):
   rows.append(dict(id=str(d['trial_id'][i]),dataset='Emotiv_external',subject='external_subject_01',session=str(d['session_id'][i]),original_split=split,label=int(y[i])))
  arrays.append(x.copy())
x=np.concatenate(arrays);assert len(rows)==880 and len({r['id'] for r in rows})==880
# Remove large device DC offset before rational resampling, avoiding edge ringing.
x=x-x.mean(-1,keepdims=True);x=resample_poly(x,25,16,axis=-1,padtype='line').astype(np.float32);x-=x.mean(-1,keepdims=True)
clip_fraction=float((np.abs(x)>1000).mean());x=np.clip(x,-1000,1000)/100
n=x.shape[-1];target=200+math.ceil((n-200)/180)*180;x=np.pad(x,((0,0),(0,0),(0,target-n)),mode='edge')
ec=json.loads(Path('configs/encoder.json').read_text());bank=AutoModel.from_pretrained(ec['positions_checkpoint'],trust_remote_code=True,local_files_only=True).eval()
lookup={canonical_channel(n):n for n in json.loads((Path(ec['positions_checkpoint'])/'config.json').read_text())['position_names']};mapped=[lookup[canonical_channel(n)] for n in channels];positions=bank(mapped).detach().cuda();assert positions.shape==(14,3)
reve,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);reve=reve.cuda().eval().requires_grad_(False)
ids=list(range(rank,len(rows),world));features=[]
with torch.no_grad():
 for start in range(0,len(ids),16):
  ix=ids[start:start+16];z=reve(torch.from_numpy(x[ix]).cuda(),positions[None].expand(len(ix),-1,-1)).flatten(1,2);assert torch.isfinite(z).all();features.append(z.half().cpu())
features=torch.cat(features);del reve,bank;torch.cuda.empty_cache()
text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,train_lora=True,train_head=True)
assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad};model.load_state_dict(saved['trainable'],strict=False);del saved
model.eval().requires_grad_(False)
if rank==0:
 write_json(a.out/'config.json',dict(checkpoint=str(checkpoint),step=2307,checkpoint_sha256=digest(checkpoint),data_hashes=files,channels=channels,original_samples=636,resampled_samples=n,padded_samples=target,tokens=int(features.shape[1]),input_fs=128,model_fs=200,clip_fraction=clip_fraction,unit='uV inferred from Cortex stream metadata and official documentation; acquisition export conversion not independently verified',unit_reference='https://emotiv.gitbook.io/cortex-api/data-subscription/data-sample-object',preprocessing='per-trial DC removal; resample_poly25/16 padtype=line; center; clip +/-1000 uV; divide100; repeat right edge to full REVE patch. No additional filter, rereference, EA, target standardization or calibration.',threshold=.5,script_sha256=digest(__file__),note='All three named splits are external evaluation only; no target optimization or checkpoint selection.'))
local=[]
with torch.no_grad():
 for start in range(0,len(ids),2):
  z=features[start:start+2].cuda();valid=torch.ones(z.shape[:2],dtype=torch.bool,device='cuda');prob=model(z,valid)['logits'].softmax(-1)[:,1].cpu().tolist()
  for i,v in zip(ids[start:start+2],prob):local.append(dict(rows[i],probability_right=v))
  if start%50==0:print('EVAL',rank,start,len(ids),flush=True)
parts=[None]*world;dist.all_gather_object(parts,local)
if rank==0:
 pred=sorted(sum(parts,[]),key=lambda r:r['id']);assert len(pred)==880
 result=dict(checkpoint_step=2307,overall=metrics(pred),by_original_split={s:metrics([r for r in pred if r['original_split']==s]) for s in ['train','validation','test']},by_session={s:metrics([r for r in pred if r['session']==s]) for s in sorted({r['session'] for r in pred})})
 write_json(a.out/'predictions.json',pred);write_json(a.out/'metrics.json',result);write_json(a.out/'complete.json',dict(n=880,elapsed_seconds=time.monotonic()-began));print('RESULT',result['overall']['accuracy'],flush=True)
dist.barrier();dist.destroy_process_group()
