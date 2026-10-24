"""Load released LEAF instruction weights and encode actual MAT windows, without training."""
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
import mne
ROOT=Path(__file__).resolve().parents[1]
LEAF_ROOT=ROOT.parent/'LEAF'
sys.path.insert(0,str(LEAF_ROOT))
from LEAF import LEAF
from load_config import build_model_config,load_yaml
import init_text_embeddings
init_text_embeddings.EMBEDDING_ROOT = LEAF_ROOT / "text_embeddings"
from init_text_embeddings import load_embeddings
from leaf_datasets.shared import pipeline

def main():
    torch.set_num_threads(3)
    ckpt=LEAF_ROOT/'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt'
    config=build_model_config(load_yaml(LEAF_ROOT/'configs/LEAF_mpnet.yaml'))
    model=LEAF(config)
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True),strict=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval().requires_grad_(False)
    text=load_embeddings(config.text_emb_model_name)
    instruction=torch.from_numpy(text['Decode mental workload states']).to(device)[None]
    prototypes=torch.from_numpy(np.stack([text['Resting'],text['Workload']])).to(device)
    records=[]
    for condition in (1,2):
        path=ROOT.parent/f'mne_data/EEGMat/Subject00_{condition}.edf'
        raw=mne.io.read_raw_edf(path,preload=True,verbose=False)
        picks=[n for n in raw.ch_names if n.startswith('EEG ') and n!='EEG A2-A1']
        names=[n.removeprefix('EEG ') for n in picks]
        raw.pick(picks).filter(.1,70,method='iir',verbose=False).resample(200,verbose=False)
        x=pipeline(raw.get_data()[:,:1000][None].astype(np.float32),names)
        with torch.inference_mode():
            emb,tokens,queries=model.forward_with_q(torch.from_numpy(x).float().to(device),instruction)
        assert torch.isfinite(emb).all() and torch.isfinite(tokens).all()
        records.append(dict(source=str(path),input_shape=list(x.shape),tower_shape=list(tokens.shape),aligned_shape=list(emb.shape),last_query_shape=list(queries[-1].shape),aligned_norm=float(emb.norm()),cosine_similarities=(emb@prototypes.T).cpu().tolist()))
    out=ROOT/'artifacts/leaf_encoder';out.mkdir(parents=True,exist_ok=True)
    result=dict(upstream_commit=subprocess.check_output(['git','-C',str(LEAF_ROOT),'rev-parse','HEAD'],text=True).strip(),checkpoint=str(ckpt),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),strict_load=True,parameters=sum(p.numel() for p in model.parameters()),instruction='Decode mental workload states',labels=['Resting','Workload'],records=records,note='Two training-subject windows: interface check only, not an accuracy or zero-shot evaluation.')
    (out/'smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
