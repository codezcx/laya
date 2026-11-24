"""Native dataset-specific cosine decisions at three instruction levels."""
import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

from .common import config, sha256, write_json
from .data import native, read_manifest, ShardArray, UPSTREAM
from LEAF import LEAF
from load_config import load_yaml, build_model_config


class EvaluationDataset(Dataset):
    def __init__(self, shards):
        self.x = ShardArray(shards)
        self.y = self.x.labels()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Author c_inference evaluates native window lengths, without 10s padding.
        return self.x[i], int(self.y[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    args = p.parse_args()
    cfg = config()
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = {k.removeprefix("model."): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    model = LEAF(build_model_config(load_yaml(UPSTREAM / "configs/LEAF_mpnet.yaml")))
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    del state
    _, _, text = native.build_prototype(native.Targets, "mpnet-base")
    rng = random.Random(cfg["seed"])
    prompts = {}
    for level in range(3):
        for ds in native.Targets:
            family = "Motor" if ds.startswith("MI") else "Emotion" if ds.startswith("EMO") else "SSVEP" if ds.startswith("SSVEP") else ds
            prompts[ds, level] = rng.choice(native.Instructions["Default" if level == 0 else family])
    manifest = read_manifest()
    result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "split": args.split,
              "scope": "full", "padding": "none, matching native direct inference", "results": [], "completed": False}
    started = time.time()
    for ds, labels in native.Targets.items():
        data = EvaluationDataset(manifest["datasets"][ds]["splits"][args.split])
        loader = DataLoader(data, batch_size=256, num_workers=4, shuffle=False)
        prototypes = torch.tensor(np.array([text[label] for label in labels]), device="cuda")
        for level in range(3):
            instruct = text[prompts[ds, level]].copy()
            if level == 2:
                mean = np.mean([text[label] for label in labels], axis=0)
                mean /= np.linalg.norm(mean)
                instruct = (instruct + mean) / 2
                instruct /= np.linalg.norm(instruct)
            instruct = torch.tensor(instruct, device="cuda")
            yy, pp = [], []
            with torch.inference_mode():
                for x, y in loader:
                    emb, _ = model(x.cuda(), instruct.unsqueeze(0).expand(len(y), -1))
                    pred = (emb @ prototypes.T).argmax(-1)
                    yy.extend(y.tolist())
                    pp.extend(pred.cpu().tolist())
            row = {"dataset": ds, "level": level, "instruction": prompts[ds, level], "windows": len(yy),
                   "accuracy": accuracy_score(yy, pp), "balanced_accuracy": balanced_accuracy_score(yy, pp), "kappa": cohen_kappa_score(yy, pp)}
            result["results"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output.parent / f"{args.output.stem}_{ds}_level{level}.npz", labels=yy, predictions=pp)
            write_json(args.output, result)
            print(row, flush=True)
    result["completed"] = True
    result["seconds"] = time.time() - started
    result["macro_bacc_by_level"] = {str(i): float(np.mean([r["balanced_accuracy"] for r in result["results"] if r["level"] == i])) for i in range(3)}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
