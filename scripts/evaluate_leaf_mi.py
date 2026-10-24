"""Frozen native LEAF and trained JEV transfer on fixed external MI subsets."""
import os,json,sys,time,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import mne
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from eeg_align.leaf_data import CONFIG,UP,digest
from eeg_align.mi_data import read_trial,canonical_channel
from eeg_align.jev import load_jev,JEVBridge
from eeg_align.train_joint import make_variant
from LEAF import LEAF
from load_config import build_model_config,load_yaml
from leaf_datasets.shared import pipeline
from leaf_datasets.electrode_unifier import ElectrodeUnifier
from init_text_embeddings import load_embeddings
OUT=Path('artifacts/leaf_2a_seed/external_mi')
DATASETS=['Cho2017','Lee2019_MI','PhysionetMI']
TASK=dict(question='Which movement is the participant imagining?',classes=[dict(class_id='left',option='Left hand motor imagery'),dict(class_id='right',option='Right hand motor imagery')])

def score(rows,prob):
 y=np.array([r['label'] for r in rows]);z=np.argmax(prob,axis=1);cm=np.zeros((2,2),int);np.add.at(cm,(y,z),1)
 return dict(n=len(y),accuracy=float((y==z).mean()),balanced_accuracy=float(np.mean(np.diag(cm)/cm.sum(1))),confusion_matrix=cm.tolist())

def main():
 rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']];torch.cuda.set_device(local);torch.set_num_threads(3);dist.init_process_group('nccl')
 if rank==0:OUT.mkdir(exist_ok=False,parents=True)
 dist.barrier();start=time.monotonic()
 manifest=Path('datasets/mi_mat/validation.jsonl');done=json.loads(Path('datasets/mi_mat/complete.json').read_text());assert digest(manifest)==done['manifest_sha256']['validation']
 rows=[json.loads(l) for l in manifest.read_text().splitlines() if json.loads(l)['dataset'] in DATASETS]
 # Entire source subject/session assigned to one rank, preserving normalization scope.
 groups=defaultdict(list)
 for r in rows:groups[(r['dataset'],r['subject'],r['session'],r['original_samples'])].append(r)
 tasks=sorted(groups.items());mine=tasks[rank::world]
 texts=load_embeddings('mpnet-base');inst=torch.from_numpy(texts['This is a motor imagery decoding task']).cuda()[None]
 proto=torch.from_numpy(np.stack([texts['Left'],texts['Right']])).cuda()
 encoder=LEAF(build_model_config(load_yaml(UP/'configs/LEAF_mpnet.yaml')))
 encoder.load_state_dict(torch.load(CONFIG['checkpoint'],map_location='cpu',weights_only=True),strict=True);encoder.cuda().eval().requires_grad_(False)
 aliases=ElectrodeUnifier('LEAF-ch65').reverse_lookup;lookup={canonical_channel(n):n for n in aliases}
 batches=[];mapping_audit={}
 for key,rr in mine:
  signals=[];names0=None
  for r in rr:
   x,y,m=read_trial(r['path'],r['dataset'],channels=None);assert y==r['label'] and str(m['subject'])==r['subject']
   keep=[i for i,n in enumerate(m['channels']) if canonical_channel(n) not in {'EOG','HEO','VEO','HEOG','VEOG','ECG','TRIGGER','TRIG','STI 014'}]
   names=[lookup[canonical_channel(m['channels'][i])] for i in keep]
   if names0 is None:names0=names
   assert names==names0
   x=x[keep]*1e-6;x=x-x.mean(0,keepdims=True);assert 200<=x.shape[1]<=2000
   signals.append(x)
  epo=mne.EpochsArray(np.stack(signals),mne.create_info(names0,200,'eeg'),verbose=False)
  # IIR avoids long FIR transients on short3s exported task epochs.
  epo.filter(.3,40,method='iir',verbose=False)
  x=pipeline(epo.get_data().astype(np.float32),names0);assert np.isfinite(x).all()
  mapping_audit[str(key)]=dict(original_channels=names0,template_locations=sorted({aliases[n] for n in names0}),samples=x.shape[-1])
  for offset in range(0,len(rr),64):
   subset=rr[offset:offset+64]
   with torch.inference_mode():
    emb,_,history=encoder.forward_with_q(torch.from_numpy(x[offset:offset+64]).float().cuda(),inst.expand(len(subset),-1))
    q=encoder.qformer.final_norm(history[-1]).cpu().numpy();prob=(emb@proto.T).softmax(-1).cpu().numpy()
   batches.append((subset,q,prob))
  (OUT/f'progress_rank{rank}.json').write_text(json.dumps(dict(stage='encode',group=str(key),encoded=sum(len(r) for r,_,_ in batches))))
 del encoder;torch.cuda.empty_cache()
 checkpoint=Path('artifacts/leaf_2a_seed/jev/best.pt');saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert saved['step']==600 and saved['config']['code_sha256']['eeg_align/jev.py']==digest('eeg_align/jev.py')
 weights=saved['trainable'];del saved
 text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,True,True,head_choices=16,input_dim=256)
 expected={n:p for n,p in model.named_parameters() if p.requires_grad};assert set(expected)==set(weights)
 with torch.no_grad():
  for n,p in expected.items():assert p.shape==weights[n].shape;p.copy_(weights[n])
 del weights;model.eval().requires_grad_(False)
 suffix={v:torch.tensor(tok.encode(make_variant(TASK,0,'',0,42,v)['suffix'],add_special_tokens=False),device='cuda') for v in [0,1]}
 result=[]
 for rr,q,native in batches:
  for offset in range(0,len(rr),16):
   r=rr[offset:offset+16];x=torch.from_numpy(q[offset:offset+16]).cuda();mask=torch.ones(x.shape[:2],dtype=torch.bool,device='cuda');ps=[]
   with torch.inference_mode():
    for v in [0,1]:
     prob=model(x,mask,suffix_ids=[suffix[v]]*len(r),num_choices=2)['logits'].softmax(-1)[:,:2]
     if v:prob=prob.flip(-1)
     ps.append(prob.cpu().numpy())
   for i,(row,p0,p1) in enumerate(zip(r,*ps)):
    result.append(dict(id=row['id'],dataset=row['dataset'],subject=row['subject'],label=row['label'],native=native[offset+i].tolist(),normal=p0.tolist(),reversed=p1.tolist()))
  (OUT/f'progress_rank{rank}.json').write_text(json.dumps(dict(stage='jev_inference',done=len(result),total=sum(len(r) for r,_,_ in batches))))
 gathered=[None]*world;dist.all_gather_object(gathered,result);allrows=sum(gathered,[]);assert len(allrows)==len(rows) and {r['id'] for r in allrows}=={r['id'] for r in rows}
 audits=[None]*world;dist.all_gather_object(audits,mapping_audit)
 if rank==0:
  report={}
  for ds in DATASETS:
   rr=[r for r in allrows if r['dataset']==ds]
   report[ds]=dict(subjects=len({r['subject'] for r in rr}),metrics={k:score(rr,np.array([r[k] for r in rr])) for k in ['native','normal','reversed']},permutation_consistency=float(np.mean([np.argmax(r['normal'])==np.argmax(r['reversed']) for r in rr])),per_subject={s:{k:score([r for r in rr if r['subject']==s],np.array([r[k] for r in rr if r['subject']==s])) for k in ['native','normal','reversed']} for s in sorted({r['subject'] for r in rr})})
  (OUT/'results.json').write_text(json.dumps(report,indent=2));(OUT/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
  (OUT/'config.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),leaf_checkpoint_sha256=digest(CONFIG['checkpoint']),manifest=str(manifest),manifest_sha256=digest(manifest),datasets=DATASETS,split='fixed previous validation subsets; no SFT',preprocessing='all mapped scalp channels;200Hz;average reference;0.3–40Hz IIR;native trial lengths;official LEAF robust scaling grouped by subject/session/length;65-channel interpolation',mapping_audits=audits,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
  (OUT/'complete.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,n=len(allrows))));print(json.dumps(report),flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
