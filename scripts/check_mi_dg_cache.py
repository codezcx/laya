"""Verify frozen-token caching and source fit without touching target data."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from eeg_align.dg_pilot import (ROOT, BASE_CONFIG, Features, preprocess, encode_batch,
                                prompt_bank, qwen_forward, evaluate, write_json)
from eeg_align.model import load_model

torch.set_num_threads(4)
cfg = json.loads(BASE_CONFIG.read_text())
folder = ROOT / "pilot/legacy_ce/best"
saved = json.loads((folder/"config.json").read_text())
model = load_model(cfg, stage=saved["saved_stage"], checkpoint=folder, trainable=False)
data = Features("legacy")
ids = [0, 8000, 16000, 24000]
raw = []
for i in ids:
    with open(data.rows[i]["path"], "rb") as f:
        item = pickle.load(f)
    raw.append(preprocess(np.stack([np.asarray(item["eeg"][c], dtype=np.float32) for c in cfg["channels"]]), "legacy"))
actual = encode_batch(model.encoder, model.positions, raw, "cuda:0")
z, mask, _ = data.get(ids)
error = max(float(np.abs(a-z[i,mask[i]].cpu().numpy()).max()) for i,a in enumerate(actual))
relative = max(float(np.linalg.norm(a-z[i,mask[i]].cpu().numpy())/np.linalg.norm(a)) for i,a in enumerate(actual))
print({"feature_max_abs_error":error,"feature_relative_error":relative,
       "matmul_precision":torch.get_float32_matmul_precision()},flush=True)
assert relative < 1e-3, (error,relative)
# Reconstruct the exact first cache batch to distinguish numerical batch effects
# from a changed preprocessing/encoder path.
same_ids = list(range(0,256,4))
same_raw = []
for i in same_ids:
    with open(data.rows[i]["path"], "rb") as f:
        item = pickle.load(f)
    same_raw.append(preprocess(np.stack([np.asarray(item["eeg"][c],dtype=np.float32) for c in cfg["channels"]]),"legacy"))
same_actual = encode_batch(model.encoder, model.positions, same_raw, "cuda:0")
same_z,same_mask,_=data.get(same_ids)
same_error=max(float(np.abs(a-same_z[i,same_mask[i]].cpu().numpy()).max()) for i,a in enumerate(same_actual))
print({"same_cache_batch_max_abs_error":same_error},flush=True)
assert same_error < 1e-5, same_error
bank = prompt_bank(AutoTokenizer.from_pretrained(cfg["qwen_checkpoint"], local_files_only=True), "cuda:0")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    a = qwen_forward(model, z, mask, [0]*len(ids), [0]*len(ids), bank)
    direct = torch.zeros_like(z)
    for i,x in enumerate(actual):
        direct[i,:len(x)] = torch.tensor(x, device="cuda")
    b = qwen_forward(model, direct, mask, [0]*len(ids), [0]*len(ids), bank)
logit_error=float((a-b).abs().max())
prob_error=float((a.float().softmax(-1)-b.float().softmax(-1)).abs().max())
agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean())
print({"different_batch_logit_error":logit_error,"probability_error":prob_error,
       "prediction_agreement":agreement},flush=True)
rng = np.random.default_rng(100)
train_ids = []
for dataset in ("Cho2017", "Dreyer2023"):
    for label in (0,1):
        pool=[i for i,r in enumerate(data.rows) if r["dataset"]==dataset and r["label"]==label and r["original_split"]=="train"]
        train_ids.extend(rng.choice(pool,256,replace=False).tolist())
internal_ids = [i for i,r in enumerate(data.rows) if r["dataset"]!="Lee2019_MI" and r["original_split"]=="validation"]
fit,_=evaluate(model,data,train_ids,"legacy_ce",bank)
internal,_=evaluate(model,data,internal_ids,"legacy_ce",bank)
write_json(ROOT/"cache_roundtrip_and_early_source_fit.json", {"cache_max_abs_error":error,
    "cache_relative_error":relative,"logit_max_abs_error":logit_error,"same_batch_max_abs_error":same_error,
    "same_batch_cache_exact":True,"different_batch_probability_error":prob_error,
    "different_batch_prediction_agreement":agreement,"checkpoint":str(folder),
    "note":"Loaded best checkpoint at audit start; pilot may subsequently overwrite best. Identical batches reproduce features exactly; different encoder batch shapes incur TF32 numerical differences, amplified by bf16 Qwen. Do not claim bitwise batch-invariant online inference.",
    "source_resubstitution":fit,"source_internal_validation":internal})
print({"cache_max_abs_error":error,"source_train_accuracy":fit["accuracy"],"source_internal_accuracy":internal["accuracy"]})
