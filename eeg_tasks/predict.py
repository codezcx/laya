"""One raw EEG window -> task-conditioned class scores. No invented task confidence."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .data import config,preprocess,normalized_window
from .model import PrototypeClassifier

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--eeg-npy',required=True);p.add_argument('--arm',default='text_sigmoid');p.add_argument('--sample-rate',type=int,default=200);p.add_argument('--task',choices=['emotion','motor_imagery']);a=p.parse_args()
 cfg=config();base=json.loads(Path(cfg['base_config']).read_text());root=Path(cfg['output']);torch.set_num_threads(4)
 raw=np.load(a.eeg_npy);assert raw.ndim==2 and raw.shape[0]==3,'Expected C3,Cz,C4 channel order'
 signal=preprocess(raw,a.sample_rate,cfg);assert signal.shape[-1]>=600
 signal=torch.from_numpy(normalized_window(signal[:,:600]))[None].cuda()
 encoder=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True).eval().requires_grad_(False).cuda();bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);pos=bank(cfg['channels']).cuda()
 stride=encoder.config.patch_size-encoder.config.patch_overlap;tail=(-(600-encoder.config.patch_size))%stride
 signal=torch.nn.functional.pad(signal,(0,tail),mode='replicate');features=encoder(signal,pos[None]).mean(2).flatten(1)
 state=torch.load(root/a.arm/'best.pt',weights_only=True,map_location='cuda');model=PrototypeClassifier(state['prototypes'],a.arm).cuda();model.load_state_dict(state);model.eval();logits=model(features)[0].float()
 result={'task_inference':'Not calibrated; task-conditioned results only','results':{}}
 for i,(start,end) in enumerate([(0,3),(3,5)]):
  task=cfg['tasks'][i]
  if a.task and a.task!=task['name']:continue
  scores=logits[start:end].softmax(-1).cpu().tolist();result['results'][task['name']]={'class':task['classes'][int(np.argmax(scores))],'conditional_scores':dict(zip(task['classes'],scores))}
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
