"""Infer one raw 200-Hz C3/Cz/C4 trial using the exact pilot preprocessing/prompt.

Input is a numeric .npy array shaped (3,T), not a pickle or label-bearing prompt.
Different GPU batch shapes can produce small TF32/bf16 numerical differences.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from eeg_align.dg_pilot import (BASE_CONFIG, SmallClassifier, preprocess, encode_batch,
                                prompt_bank, qwen_forward, QUESTIONS)
from eeg_align.model import load_model

p=argparse.ArgumentParser()
p.add_argument("--run",required=True,help="e.g. artifacts/mi_zero_shot_v1/refit/legacy_ce")
p.add_argument("--eeg-npy",required=True,help="C3,Cz,C4 in that order, 200 Hz, consistent amplitude units")
p.add_argument("--question",choices=("mi","mu","beta"),default="mi")
args=p.parse_args()
run=Path(args.run); info=json.loads((run/"run.json").read_text()); arm=info["args"]["arm"]
cfg=info["config"]; mode=info["normalization"]
raw=np.load(args.eeg_npy,allow_pickle=False)
if raw.ndim!=2 or raw.shape[0]!=3 or raw.shape[1]<600 or not np.isfinite(raw).all():
    raise ValueError("Expected finite 3-channel signal with at least 600 samples")
task=("mi","mu","beta").index(args.question)
if task and arm!="grounded_qa":
    raise ValueError("Only grounded_qa was trained on spectral questions")
torch.set_num_threads(4)
small=arm.startswith("reve_mlp")
if small:
    model=SmallClassifier().cuda().eval()
    model.load_state_dict(torch.load(run/"best.pt",weights_only=True))
    encoder=AutoModel.from_pretrained(cfg["reve_checkpoint"],trust_remote_code=True,local_files_only=True).cuda().eval()
    positions=AutoModel.from_pretrained(cfg["positions_checkpoint"],trust_remote_code=True,local_files_only=True)(cfg["position_names"]).detach().cuda()
else:
    saved=json.loads((run/"best/config.json").read_text())
    model=load_model(cfg,checkpoint=run/"best",stage=saved["saved_stage"],trainable=False)
    encoder,positions=model.encoder,model.positions
    bank=prompt_bank(AutoTokenizer.from_pretrained(cfg["qwen_checkpoint"],local_files_only=True),"cuda:0")
with torch.inference_mode():
    features=encode_batch(encoder,positions,[preprocess(raw.astype("float32"),mode)],"cuda:0")[0]
    z=torch.from_numpy(features[None]).cuda()
    mask=torch.ones(z.shape[:2],device="cuda",dtype=torch.bool)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(z,mask) if small else qwen_forward(model,z,mask,[task],[0],bank)
    probabilities=logits.float().softmax(-1)[0].cpu().tolist()
    meanings=QUESTIONS[task][1:]
    result={"question":QUESTIONS[task][0],"prediction":meanings[int(np.argmax(probabilities))],
            "probabilities":dict(zip(meanings,probabilities)),"eeg_tokens":len(features),
            "preprocessing":mode,"checkpoint":str(run),"autoregressive_generation":False}
print(json.dumps(result,indent=2))
