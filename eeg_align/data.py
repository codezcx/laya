"""Trusted local MOABB pickles -> three-channel raw EEG; deterministic session splits."""
import argparse
import json
import pickle
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def preprocessing_signature(config):
    keys = ("channels", "sample_rate", "normalization", "labels", "tail_padding")
    return hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def read_trial(path, config):
    # Pickle input is limited to the trusted mounted dataset, not untrusted uploads.
    with open(path, "rb") as f:
        item = pickle.load(f)
    if item["freq"] != config["sample_rate"] or item["paradigm"] != "imagery":
        raise ValueError(f"Wrong sample rate/paradigm: {path}")
    label = config["labels"].index(item["label"])
    eeg = np.stack([np.asarray(item["eeg"][c], dtype=np.float32) for c in config["channels"]])
    if eeg.ndim != 2 or eeg.shape[1] < 200 or not np.isfinite(eeg).all():
        raise ValueError(f"Invalid EEG: {path}")
    if config["normalization"] != "per_channel_zscore_clip15":
        raise ValueError("Unsupported preprocessing")
    # Trial-local, no test-population statistics. This intentionally removes absolute amplitude.
    std = eeg.std(axis=-1, keepdims=True)
    if np.any(std < 1e-8):
        raise ValueError(f"Flat electrode: {path}")
    eeg = np.clip((eeg - eeg.mean(axis=-1, keepdims=True)) / std, -15, 15)
    return torch.from_numpy(eeg.copy()), label, item


class MIDataset(Dataset):
    def __init__(self, manifest, config):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.config = config
        self.caches = {}
        signature = preprocessing_signature(config)
        for row in self.rows:
            if "cache_file" in row and row.get("cache_signature") != signature:
                raise ValueError("Cached EEG preprocessing does not match config")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if "cache_file" in row:
            path = row["cache_file"]
            if path not in self.caches:
                self.caches[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            eeg = torch.from_numpy(self.caches[path][row["cache_index"], :, :row["length"]].copy())
            return {"eeg": eeg, "label": row["label"], "path": row["path"],
                    "dataset_index": row["dataset_index"]}
        eeg, label, item = read_trial(row["path"], self.config)
        if label != row["label"] or str(item["subject"]) != str(row["subject"]) or item["session"] != row["session"]:
            raise ValueError(f"Manifest identity mismatch: {row['path']}")
        return {"eeg": eeg, "label": label, "path": row["path"], "dataset_index": row.get("dataset_index", 0)}


def collate(rows):
    lengths = torch.tensor([r["eeg"].shape[1] for r in rows])
    eeg = torch.zeros(len(rows), rows[0]["eeg"].shape[0], int(lengths.max()))
    for i, row in enumerate(rows):
        eeg[i, :, :lengths[i]] = row["eeg"]
    return {"eeg": eeg, "lengths": lengths, "labels": torch.tensor([r["label"] for r in rows]),
            "dataset_index": torch.tensor([r.get("dataset_index", 0) for r in rows])}


class BalancedDistributedSampler(Sampler):
    """Dataset/class-balanced replacement draws, globally seeded then sharded by rank."""
    def __init__(self, rows, rank=0, world_size=1, seed=42):
        groups = [(r.get("dataset", "default"), r["label"]) for r in rows]
        counts = Counter(groups)
        self.weights = torch.tensor([1. / counts[g] for g in groups], dtype=torch.double)
        self.rank, self.world, self.seed, self.epoch = rank, world_size, seed, 0
        self.num_samples = (len(rows) + world_size - 1) // world_size

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(self.weights, self.num_samples*self.world, replacement=True, generator=generator)
        return iter(draws[self.rank::self.world].tolist())


def prepare(config):
    root = Path(config["data_root"])
    if root.name != "BNCI2014_004":
        raise ValueError("This split builder is only for BNCI2014_004; provide audited manifests for other datasets")
    splits = {s: [] for s in ("train", "validation", "test")}
    unresolved = 0
    for path in sorted(root.glob("subject_*/session_*/*.pkl")):
        _, label, item = read_trial(path, config)
        if path.parent.name != "session_" + str(item["session"]) or path.parent.parent.name != "subject_" + str(item["subject"]):
            raise ValueError(f"Directory/metadata mismatch: {path}")
        session = item["session"]
        split = {"0train": "train", "1train": "train", "2train": "validation",
                 "3test": "test", "4test": "test"}[session]
        unresolved += int(item.get("source_resolution") == "inferred_file_by_run")
        splits[split].append({"path": str(path.resolve()), "label": label, "subject": item["subject"],
                              "session": session, "source_resolution": item.get("source_resolution")})
    if any(not rows for rows in splits.values()):
        raise ValueError("Missing split")
    out = Path(config["manifest_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    report = {"splits": {s: {"count": len(rows), "labels": dict(Counter(r["label"] for r in rows))}
                         for s, rows in splits.items()},
              "protocol": "Within-subject held-out session; not unseen-subject evaluation",
              "provenance_verified": False, "inferred_source_rows": unresolved,
              "note": "raw_name/source mapping requires verification against originals before formal experiments; 2b has bipolar leads"}
    (out / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reve_qwen_mi.json")
    args = parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()))
