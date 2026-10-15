"""Source-only frequency features aligned to the same spatial anchors as REVE."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .source_probe import spectral_features
from .spatial_model import spatial_pool


def fit_standardizer(features,train_mask):
    if not train_mask.any():raise ValueError('Training rows required')
    mean=features[train_mask].mean(0,keepdim=True)
    std=features[train_mask].std(0,unbiased=False,keepdim=True).clamp_min(1e-5)
    return mean,std

@torch.no_grad()
def prepare_spectral(rows,device,cfg):
    base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);anchors=bank(cfg['anchors']).to(device)
    root=Path('artifacts/multitask_prototypes_v1');allrows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()];lookup={r['signal_sha256']:i for i,r in enumerate(allrows)}
    eroot=Path('artifacts/emotion_source_probe_v4');erows=[json.loads(s) for s in (eroot/'rows.jsonl').read_text().splitlines()];elookup={r['signal_sha256']:i for i,r in enumerate(erows)};e=np.load(eroot/'62ch_relative_30.npy').reshape(-1,62,5)
    prep=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());allnames=[s.split()[-1] for s in Path(prep['seed_channel_file']).read_text().splitlines() if s.strip()];names=json.loads(Path('artifacts/emotion_reve60_v5/0.done.json').read_text())['channels'];picks=[allnames.index(c) for c in names];raw=np.load(root/'signals.npy',mmap_mode='r')
    output=torch.zeros(len(rows),len(anchors),5,device=device)
    for task in (0,1):
        ids=[i for i,r in enumerate(rows) if r['task']==task]
        if task==0:
            feature=e[[elookup[rows[i]['signal_sha256']] for i in ids]][:,picks];pos=bank(names).to(device)
        else:
            signal=np.array(raw[[lookup[rows[i]['signal_sha256']] for i in ids]])
            feature=spectral_features(signal)[:,:15].reshape(-1,3,5);pos=bank(['C3','Cz','C4']).to(device)
        output[ids]=spatial_pool(torch.from_numpy(feature).to(device),pos,anchors,cfg['sigma'])
    mask=torch.tensor([r['split']=='train' for r in rows],device=device);mean,std=fit_standardizer(output,mask)
    assert torch.isfinite(output).all()
    return (output-mean)/std,{'mean':mean.cpu().tolist(),'std':std.cpu().tolist(),'training_rows':int(mask.sum()),'bands_hz':[[1,4],[4,8],[8,13],[13,20],[20,30]],'note':'SEED: log power minus log window variance before clip; MI: power of previously zscored clip15 input. Source train-only standardization.'}
