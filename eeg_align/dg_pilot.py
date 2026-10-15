"""Source-only domain-generalization pilot; frozen REVE cache and controlled Qwen arms.

Target evaluation is a separate command, gated by a persisted selection file.
All signal statistics are trial-local. No target fitting, adaptation or label-derived text.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, welch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .model import load_model

ROOT = Path("artifacts/mi_zero_shot_v1")
BASE_CONFIG = Path("configs/reve_qwen_mi_multi.json")
MODES = ("legacy", "harmonized")
QUESTIONS = [
    ("Which hand is the participant imagining moving?", "left hand", "right hand"),
    ("Which channel has higher mean power spectral density in 8-13 Hz?", "C3", "C4"),
    ("Which channel has higher mean power spectral density in 13-30 Hz?", "C3", "C4"),
]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def rows_for(split):
    cfg = json.loads(BASE_CONFIG.read_text())
    return [json.loads(s) for s in (Path(cfg["manifest_dir"]) / f"{split}.jsonl").read_text().splitlines()]


def source_rows():
    rows = []
    for split in ("train", "validation", "test"):
        for row in rows_for(split):
            if split == "test" and row["dataset"] != "Lee2019_MI":
                continue
            rows.append({**row, "original_split": split})
    assert all(r["dataset"] in ("Cho2017", "Lee2019_MI", "Dreyer2023") for r in rows)
    assert len({r["path"] for r in rows}) == len(rows)
    return rows


def preprocess(raw, mode):
    raw = np.asarray(raw, dtype=np.float64)
    if mode == "legacy":
        x = raw - raw.mean(-1, keepdims=True)
        return np.clip(x / np.maximum(x.std(-1, keepdims=True), 1e-8), -15, 15).astype("float32")
    if mode != "harmonized" or raw.shape[-1] < 600:
        raise ValueError((mode, raw.shape))
    # Filter full trial before cropping to limit edge effects, then retain first 3 s.
    x = sosfiltfilt(butter(4, [4, 38], btype="bandpass", fs=200, output="sos"), raw, axis=-1)[:, :600]
    x = x - x.mean(-1, keepdims=True)
    # One scale shared by all electrodes retains C3/C4 relative power.
    return np.clip(x / max(float(np.sqrt(np.mean(x*x))), 1e-8), -15, 15).astype("float32")


def signal_answers(x):
    f, psd = welch(x, fs=200, nperseg=200, noverlap=100, axis=-1)
    answers, margins = [], []
    for lo, hi in ((8, 13), (13, 30)):
        power = psd[:, (f >= lo) & (f < hi)].mean(-1)
        answers.append(int(power[2] > power[0]))
        margins.append(float(abs(np.log((power[2] + 1e-12) / (power[0] + 1e-12)))))
    return answers, margins


@torch.no_grad()
def encode_batch(encoder, positions, signals, device):
    result = [None] * len(signals)
    for n in sorted({x.shape[-1] for x in signals}):
        ix = [i for i, x in enumerate(signals) if x.shape[-1] == n]
        x = torch.from_numpy(np.stack([signals[i] for i in ix])).to(device)
        patch = encoder.config.patch_size
        stride = patch - encoder.config.patch_overlap
        tail = (-(n - patch)) % stride
        if tail:
            x = F.pad(x, (0, tail), mode="replicate")
        z = encoder(x, positions[None].expand(len(ix), -1, -1)).flatten(1, 2).float().cpu().numpy()
        for i, value in zip(ix, z):
            result[i] = value
    return result


def cache(args):
    cfg = json.loads(BASE_CONFIG.read_text())
    out = ROOT / ("target_cache" if args.target else "source_cache")
    out.mkdir(parents=True, exist_ok=True)
    if args.target and not (ROOT / "selection.json").exists():
        raise ValueError("Freeze source-only selection before accessing targets")
    rows = ([{**r, "original_split": s} for s in ("external_2a", "external_2b") for r in rows_for(s)]
            if args.target else source_rows())
    indices = list(range(args.shard, len(rows), args.shards))
    done = out / f"shard_{args.shard}.done.json"
    if done.exists():
        raise FileExistsError(done)
    torch.set_num_threads(4)
    device = "cuda:0"
    encoder, info = AutoModel.from_pretrained(cfg["reve_checkpoint"], local_files_only=True,
        trust_remote_code=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(info)
    encoder = encoder.eval().requires_grad_(False).to(device)
    bank = AutoModel.from_pretrained(cfg["positions_checkpoint"], local_files_only=True, trust_remote_code=True)
    positions = bank(cfg["position_names"]).detach().to(device)
    arrays = {m: np.lib.format.open_memmap(out/f"{m}_{args.shard}.npy", mode="w+", dtype="float32",
              shape=(len(indices), 18, 512)) for m in MODES}
    metadata = []
    start = time.monotonic()
    for begin in range(0, len(indices), 64):
        ids = indices[begin:begin+64]
        raw = []
        for idx in ids:
            row = rows[idx]
            with open(row["path"], "rb") as f:
                item = pickle.load(f)
            assert item["freq"] == 200 and item["paradigm"] == "imagery"
            assert cfg["labels"].index(item["label"]) == row["label"]
            assert str(item["subject"]) == str(row["subject"])
            x = np.stack([np.asarray(item["eeg"][c], dtype=np.float32) for c in cfg["channels"]])
            assert np.isfinite(x).all() and (x.std(-1) > 1e-8).all()
            raw.append(x)
        transformed = {m: [preprocess(x, m) for x in raw] for m in MODES}
        encoded = {m: encode_batch(encoder, positions, transformed[m], device) for m in MODES}
        for j, idx in enumerate(ids):
            row = {**rows[idx], "global_index": idx, "shard": args.shard, "local_index": begin+j}
            row["token_lengths"] = {}
            for m in MODES:
                z = encoded[m][j]
                if len(z) > 18 or not np.isfinite(z).all():
                    raise ValueError("Invalid cached features")
                arrays[m][begin+j] = 0
                arrays[m][begin+j, :len(z)] = z
                row["token_lengths"][m] = len(z)
            row["aux_labels"], row["aux_margins"] = signal_answers(transformed["harmonized"][j])
            metadata.append(row)
        if begin % 640 == 0:
            print(json.dumps({"cached": begin+len(ids), "total": len(indices), "elapsed": time.monotonic()-start}), flush=True)
    for x in arrays.values():
        x.flush()
    (out/f"rows_{args.shard}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in metadata))
    write_json(done, {"n": len(indices), "seconds": time.monotonic()-start, "dtype": "float32",
        "encoder": cfg["reve_checkpoint"], "source_only": not args.target,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


class Features:
    def __init__(self, mode, target=False):
        self.mode = mode
        folder = ROOT / ("target_cache" if target else "source_cache")
        done = sorted(folder.glob("shard_*.done.json"))
        if len(done) != 4:
            raise ValueError("Require all four complete cache shards")
        self.arrays = {i: np.load(folder/f"{mode}_{i}.npy", mmap_mode="r") for i in range(4)}
        self.rows = sorted([json.loads(s) for i in range(4) for s in (folder/f"rows_{i}.jsonl").read_text().splitlines()],
                           key=lambda r: r["global_index"])
        if [r["global_index"] for r in self.rows] != list(range(len(self.rows))):
            raise ValueError("Cache coverage mismatch")

    def get(self, ids, device="cuda:0"):
        rows = [self.rows[int(i)] for i in ids]
        features = torch.from_numpy(np.stack([self.arrays[r["shard"]][r["local_index"]] for r in rows])).to(device)
        lengths = torch.tensor([r["token_lengths"][self.mode] for r in rows], device=device)
        mask = torch.arange(18, device=device)[None] < lengths[:, None]
        labels = torch.tensor([r["label"] for r in rows], device=device)
        return features, mask, labels


def prompt_bank(tokenizer, device):
    bank = {}
    for task, (question, a, b) in enumerate(QUESTIONS):
        for swap in (0, 1):
            options = (b, a) if swap else (a, b)
            rendered = tokenizer.apply_chat_template([
                {"role": "system", "content": "Answer the question about the EEG. Select A or B."},
                {"role": "user", "content": "EEG channels: C3, Cz, C4. Sampling rate: 200 Hz.\nEEG begins:\n__EEG__\nEEG ends.\n" +
                 question + f"\nOptions: A. {options[0]}; B. {options[1]}.\nAnswer:"}],
                tokenize=False, add_generation_prompt=True)
            prefix, suffix = rendered.split("__EEG__")
            bank[(task, swap)] = [torch.tensor(tokenizer.encode(t, add_special_tokens=False), device=device) for t in (prefix, suffix)]
    return bank


def qwen_forward(model, features, mask, tasks, swaps, bank, legacy_prompt=False):
    z = model.connector(features)
    embed = model.text.get_input_embeddings()
    seqs = []
    for i, (task, swap) in enumerate(zip(tasks, swaps)):
        ids = (model.prefix_ids, model.suffix_ids) if legacy_prompt else bank[(int(task), int(swap))]
        pre, post = [embed(p) for p in ids]
        seqs.append(torch.cat([pre, z[i, mask[i]].to(pre.dtype), post], dim=0))
    lengths = torch.tensor([len(s) for s in seqs], device=z.device)
    inputs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    attention = torch.arange(inputs.shape[1], device=z.device)[None] < lengths[:, None]
    positions = (attention.long().cumsum(-1)-1).clamp_min(0)
    hidden = model.text(inputs_embeds=inputs, attention_mask=attention, position_ids=positions,
                        use_cache=False, return_dict=True).last_hidden_state
    state = hidden[torch.arange(len(hidden), device=z.device), lengths-1]
    with torch.autocast("cuda", enabled=False):
        return model.head(state.float())


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 256), nn.GELU(),
                                 nn.Dropout(.2), nn.Linear(256, 2))

    def forward(self, z, mask):
        # Retain electrode identities: temporal pooling within each electrode.
        rows = [x[v].reshape(3, -1, 512).mean(1).flatten() for x, v in zip(z, mask)]
        return self.net(torch.stack(rows))


def metrics(rows, probabilities):
    y = np.array([r["label"] for r in rows])
    pred = np.asarray(probabilities).argmax(-1)
    subjects = sorted({(r["dataset"], str(r["subject"])) for r in rows})
    by_subject = {}
    for dataset, subject in subjects:
        ids = [i for i, r in enumerate(rows) if r["dataset"] == dataset and str(r["subject"]) == subject]
        by_subject[f"{dataset}/{subject}"] = {"n": len(ids), "accuracy": float(np.mean(pred[ids] == y[ids]))}
    cm = np.bincount(2*y+pred, minlength=4).reshape(2, 2)
    accs = np.array([r["accuracy"] for r in by_subject.values()])
    boot = np.random.default_rng(2026).choice(accs, size=(5000, len(accs)), replace=True).mean(1)
    return {"n": len(y), "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": float((cm.diagonal()/np.maximum(cm.sum(1), 1)).mean()),
            "majority_baseline_accuracy": float(np.bincount(y, minlength=2).max()/len(y)),
            "subject_macro_accuracy": float(accs.mean()),
            "subject_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(), "confusion_matrix": cm.tolist(),
            "per_subject": by_subject}


@torch.no_grad()
def evaluate(model, data, ids, arm, bank=None, swap=0, perturb=None, task=0):
    model.eval()
    probs = []
    input_ids = list(ids)
    if perturb == "shuffle":
        input_ids = np.random.default_rng(723).permutation(input_ids).tolist()
    for begin in range(0, len(ids), 32):
        batch_ids = input_ids[begin:begin+32]
        z, mask, _ = data.get(batch_ids)
        if perturb == "zero":
            z = torch.zeros_like(z)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = (model(z, mask) if arm.startswith("reve_mlp") else
                      qwen_forward(model, z, mask, [task]*len(z), [swap]*len(z), bank))
        p = logits.float().softmax(-1).cpu().numpy()
        probs.extend(p[:, ::-1] if swap else p)
    rows = [dict(data.rows[i]) for i in ids]
    if task:
        for r in rows:
            r["label"] = r["aux_labels"][task-1]
    return metrics(rows, probs), np.asarray(probs)


def train(args):
    out = ROOT / ("refit" if args.refit else "pilot") / args.arm
    out.mkdir(parents=True, exist_ok=True)
    if (out/"run.json").exists():
        raise FileExistsError(out)
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    cfg = json.loads(BASE_CONFIG.read_text())
    mode = "legacy" if args.arm in ("legacy_ce", "reve_mlp_legacy") else "harmonized"
    data = Features(mode)
    train_ids = [i for i, r in enumerate(data.rows) if r["original_split"] == "train" and
                 (args.refit or r["dataset"] != "Lee2019_MI")]
    val_ids = [i for i, r in enumerate(data.rows) if r["dataset"] == "Lee2019_MI" and r["original_split"] != "train"]
    assert not set(train_ids) & set(val_ids)
    assert not ({(data.rows[i]["dataset"], data.rows[i]["subject"]) for i in train_ids} &
                {(data.rows[i]["dataset"], data.rows[i]["subject"]) for i in val_ids})
    if not args.refit:
        assert all(data.rows[i]["dataset"] != "Lee2019_MI" for i in train_ids)
    write_json(out/"run.json", {"args": vars(args), "config": cfg, "train_n": len(train_ids),
        "validation_n": len(val_ids), "selection": "Lee2019_MI held-out-subject macro accuracy",
        "normalization": mode, "seed": seed, "torch": torch.__version__,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    is_mlp = args.arm.startswith("reve_mlp")
    model = SmallClassifier().cuda() if is_mlp else load_model(cfg, stage="warmup")
    # Feature cache is from this exact frozen encoder; release duplicate GPU copy.
    if not is_mlp:
        model.encoder.cpu()
        bank = prompt_bank(AutoTokenizer.from_pretrained(cfg["qwen_checkpoint"], local_files_only=True), "cuda:0")
    else:
        bank = None
    groups = ([{"params": model.parameters(), "lr": 1e-3}] if is_mlp else
              [{"params": model.connector.parameters(), "lr": 1e-4}, {"params": model.head.parameters(), "lr": 1e-4}])
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    groups_count = Counter((data.rows[i]["dataset"], data.rows[i]["label"]) for i in train_ids)
    weights = torch.tensor([1/groups_count[(data.rows[i]["dataset"], data.rows[i]["label"])] for i in train_ids], dtype=torch.double)
    sampler_gen = torch.Generator().manual_seed(seed)
    draws = torch.multinomial(weights, args.steps*32, replacement=True, generator=sampler_gen).numpy()
    rng = np.random.default_rng(seed+1)
    best = -1
    started = time.monotonic()
    for step in range(1, args.steps+1):
        if not is_mlp and step == 81:
            model.enable_lora()
            model.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            optimizer.add_param_group({"params": [p for p in model.text.parameters() if p.requires_grad], "lr": 1e-5})
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, aux_losses = [], []
        for micro in range(4):
            loc = (step-1)*32 + micro*8
            ids = [train_ids[int(i)] for i in draws[loc:loc+8]]
            z, mask, labels = data.get(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if is_mlp:
                    logits = model(z, mask)
                    loss = F.cross_entropy(logits.float(), labels)
                else:
                    swaps = rng.integers(0, 2, len(ids)) if args.arm == "grounded_qa" else np.zeros(len(ids), dtype=int)
                    logits = qwen_forward(model, z, mask, [0]*len(ids), swaps, bank)
                    loss = F.cross_entropy(logits, labels ^ torch.tensor(swaps, device="cuda"))
            losses.append(float(loss.detach()))
            (loss/4).backward()
            if args.arm == "grounded_qa":
                tasks = rng.integers(1, 3, len(ids))
                swaps = rng.integers(0, 2, len(ids))
                labels_aux = torch.tensor([data.rows[i]["aux_labels"][t-1] ^ int(s) for i,t,s in zip(ids,tasks,swaps)], device="cuda")
                valid = torch.tensor([data.rows[i]["aux_margins"][t-1] >= .1 for i,t in zip(ids,tasks)], device="cuda")
                if valid.any():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits_aux = qwen_forward(model, z, mask, tasks, swaps, bank)
                        loss_aux = F.cross_entropy(logits_aux[valid], labels_aux[valid])
                    (.25*loss_aux/4).backward()
                    aux_losses.append(float(loss_aux.detach()))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step % 10 == 0 or step == 1:
            log = {"step": step, "mi_loss": float(np.mean(losses)), "aux_loss": float(np.mean(aux_losses)) if aux_losses else None,
                   "seconds": time.monotonic()-started, "peak_gpu_gb": torch.cuda.max_memory_allocated()/1e9}
            with open(out/"train.jsonl", "a") as f:
                f.write(json.dumps(log)+"\n")
            print(json.dumps(log), flush=True)
        if (step % 100 == 0 or step == args.steps) and not args.refit:
            result, probabilities = evaluate(model, data, val_ids, args.arm, bank)
            result.update(step=step, seconds=time.monotonic()-started)
            with open(out/"validation.jsonl", "a") as f:
                f.write(json.dumps(result)+"\n")
            print(json.dumps({"validation": result}), flush=True)
            if result["subject_macro_accuracy"] > best:
                best = result["subject_macro_accuracy"]
                if is_mlp:
                    torch.save(model.state_dict(), out/"best.pt")
                else:
                    model.save(out/"best")
                write_json(out/"best_metrics.json", result)
                np.save(out/"validation_probabilities.npy", probabilities)
    if args.refit:
        if is_mlp:
            torch.save(model.state_dict(), out/"best.pt")
        else:
            model.save(out/"best")
    write_json(out/"complete.json", {"steps": args.steps, "seconds": time.monotonic()-started,
                                    "peak_gpu_gb": torch.cuda.max_memory_allocated()/1e9})


def test(args):
    selection = json.loads((ROOT/"selection.json").read_text())
    if args.arm not in selection["frozen_arms"]:
        raise ValueError("Arm not frozen before target evaluation")
    folder = ROOT / ("refit" if args.refit else "pilot") / args.arm
    if not (folder/"complete.json").exists():
        raise ValueError("Training must complete")
    out = folder/"evaluation"
    out.mkdir(exist_ok=False)
    cfg = json.loads(BASE_CONFIG.read_text())
    mode = "legacy" if args.arm in ("legacy_ce", "reve_mlp_legacy") else "harmonized"
    bank = None
    if args.arm.startswith("reve_mlp"):
        model = SmallClassifier().cuda()
        model.load_state_dict(torch.load(folder/"best.pt", weights_only=True))
    else:
        saved = json.loads((folder/"best/config.json").read_text())
        model = load_model(cfg, stage=saved["saved_stage"], checkpoint=folder/"best", trainable=False)
        model.encoder.cpu()
        bank = prompt_bank(AutoTokenizer.from_pretrained(cfg["qwen_checkpoint"], local_files_only=True), "cuda:0")
    torch.set_num_threads(4)
    data = Features(mode, target=True)
    for split in ("external_2a", "external_2b"):
        ids = [i for i,r in enumerate(data.rows) if r["original_split"] == split]
        result, probs = evaluate(model, data, ids, args.arm, bank)
        write_json(out/f"{split}.json", result)
        np.save(out/f"{split}_probabilities.npy", probs)
        (out/f"{split}_rows.jsonl").write_text("".join(json.dumps({k:data.rows[i][k] for k in ("path","label","subject","session","dataset")})+"\n" for i in ids))
        print(json.dumps({"split": split, **result}), flush=True)
    # Diagnostics on source-only proxy; target results do not influence these conditions.
    source = Features(mode)
    ids = [i for i,r in enumerate(source.rows) if r["dataset"] == "Lee2019_MI" and r["original_split"] != "train"]
    diagnostics = {}
    internal = [i for i,r in enumerate(source.rows) if r["dataset"] != "Lee2019_MI" and r["original_split"] == "validation"]
    diagnostics["source_internal_validation"], _ = evaluate(model, source, internal, args.arm, bank)
    diagnostics["source_proxy_normal"], _ = evaluate(model, source, ids, args.arm, bank)
    for perturb in ("zero", "shuffle"):
        diagnostics[perturb], _ = evaluate(model, source, ids, args.arm, bank, perturb=perturb)
    if args.arm == "grounded_qa":
        diagnostics["option_swap"], _ = evaluate(model, source, ids, args.arm, bank, swap=1)
    if args.arm == "grounded_qa":
        for task in (1, 2):
            diagnostics[f"aux_task_{task}"], _ = evaluate(model, source, ids, args.arm, bank, task=task)
            diagnostics[f"aux_task_{task}_shuffle"], _ = evaluate(model, source, ids, args.arm, bank, task=task, perturb="shuffle")
    write_json(out/"source_diagnostics.json", diagnostics)
    write_json(out/"complete.json", {"selection_sha256": hashlib.sha256((ROOT/"selection.json").read_bytes()).hexdigest()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("cache", "train", "test"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--target", action="store_true")
    p.add_argument("--arm", choices=("legacy_ce", "harmonized_ce", "grounded_qa", "reve_mlp", "reve_mlp_legacy"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--refit", action="store_true")
    args = p.parse_args()
    {"cache": cache, "train": train, "test": test}[args.command](args)


if __name__ == "__main__":
    main()
