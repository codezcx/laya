"""REVE-only MI baselines. No Qwen, connector, image encoder, or LoRA.

Time-mean pooling retains channel identity: B,C,P,D -> B,C*D -> Linear(.,2).
Model code is loaded only with explicit local-code authorization by the caller.
"""
import contextlib
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import MIDataset, preprocessing_signature


SPLITS = ("train", "validation", "test", "external_2a", "external_2b")
EXPECTED_COUNTS = dict(zip(SPLITS, (33272, 4160, 4200, 1296, 2840)))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifests(config, reference_hashes):
    """Check exact prior split files, identities, signal overlap and cache signature."""
    root = Path(config["manifest_dir"])
    reference = json.loads(Path(reference_hashes).read_text())
    by_name = {Path(k).name: v for k, v in reference.items()}
    audit = json.loads((root / "audit.json").read_text())
    if not audit.get("ready_for_exploratory_training"):
        raise ValueError("Dataset audit does not permit exploratory training")
    result = {"splits": {}, "hashes": {}, "provenance_verified": audit["provenance_verified"],
              "limitations": audit["provenance_note"]}
    identities = {}
    for name in (*SPLITS, "audit", "subject_split"):
        suffix = ".jsonl" if name in SPLITS else ".json"
        path = root / (name + suffix)
        value = file_sha256(path)
        if value != by_name.get(path.name):
            raise ValueError(f"Changed prior experiment manifest: {path}")
        result["hashes"][path.name] = value
        if name not in SPLITS:
            continue
        ds = MIDataset(path, config)
        if len(ds) != EXPECTED_COUNTS[name]:
            raise ValueError(f"Unexpected split size: {name} {len(ds)}")
        paths = {r["path"] for r in ds.rows}
        signals = {r["signal_sha256"] for r in ds.rows}
        if len(paths) != len(ds) or len(signals) != len(ds):
            raise ValueError(f"Repeated path/signal within {name}")
        people = {(r["dataset"], str(r["subject"])) for r in ds.rows}
        identities[name] = {"paths": paths, "signals": signals, "people": people}
        result["splits"][name] = {"n": len(ds), "datasets": dict(Counter(r["dataset"] for r in ds.rows)),
                                  "n_subjects": len(people)}
        for row in ds.rows:
            if config["dataset_names"][row["dataset_index"]] != row["dataset"]:
                raise ValueError("Dataset index/name mismatch")
            if row["label"] not in (0, 1) or row["cache_signature"] != preprocessing_signature(config):
                raise ValueError("Invalid label/preprocessing signature")
    for left, right in itertools.combinations(SPLITS, 2):
        for kind in ("paths", "signals", "people"):
            if identities[left][kind] & identities[right][kind]:
                raise ValueError(f"Split overlap: {left}/{right}/{kind}")
    result["all_split_identity_checks_passed"] = True
    return result


def pool_channels(encoded):
    if encoded.ndim != 4:
        raise ValueError("Expected REVE output B,C,P,D")
    return encoded.mean(dim=2).flatten(1)


class ReveClassifier(nn.Module):
    def __init__(self, encoder, positions, mode):
        super().__init__()
        if mode not in ("linear_probe", "full_sft"):
            raise ValueError(mode)
        self.encoder = encoder
        self.mode = mode
        self.register_buffer("positions", positions.clone().float())
        self.head = nn.Linear(len(positions) * encoder.config.embed_dim, 2)
        self.encoder.requires_grad_(mode == "full_sft")
        # The exported encoder has a query used only by its optional attention_pooling
        # method. This baseline uses time means, so this otherwise unused parameter
        # is not optimized. Every parameter on the encoder forward path is unfrozen.
        if hasattr(self.encoder, "cls_query_token"):
            self.encoder.cls_query_token.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "linear_probe":
            self.encoder.eval()
        return self

    def encode(self, eeg, lengths):
        patch = self.encoder.config.patch_size
        stride = patch - self.encoder.config.patch_overlap
        if lengths.shape != (len(eeg),) or torch.any(lengths > eeg.shape[-1]):
            raise ValueError("Invalid true lengths")
        pieces, order = [], []
        grad = torch.no_grad() if self.mode == "linear_probe" else contextlib.nullcontext()
        with grad, torch.autocast(device_type=eeg.device.type, enabled=False):
            # Never expose padding from other examples to REVE attention.
            for n in lengths.unique(sorted=True).tolist():
                if n < patch:
                    raise ValueError("EEG shorter than one REVE patch")
                ids = (lengths == n).nonzero().flatten()
                x = eeg[ids, :, :n].float()
                tail = (-(n - patch)) % stride
                if tail:
                    x = F.pad(x, (0, tail), mode="replicate")
                encoded = self.encoder(x, self.positions[None].expand(len(ids), -1, -1))
                pieces.append(pool_channels(encoded))
                order.append(ids)
        return torch.cat(pieces)[torch.cat(order).argsort()]

    def forward(self, eeg=None, lengths=None, features=None, labels=None):
        if (features is None) == (eeg is None):
            raise ValueError("Supply exactly one of EEG or cached features")
        if features is not None and self.mode != "linear_probe":
            raise ValueError("Full fine-tuning cannot use cached encoder features")
        features = self.encode(eeg, lengths) if features is None else features
        logits = self.head(features.float())
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


def load_reve_classifier(config, mode, device, *, allow_local_reve_code=False):
    if not allow_local_reve_code:
        raise PermissionError("Explicit authorization is required to execute local REVE model code")
    from transformers import AutoModel
    paths = [Path(config[key]).resolve() for key in ("reve_checkpoint", "positions_checkpoint")]
    if not all(p.is_dir() for p in paths):
        raise ValueError("Only existing local REVE snapshots may be loaded")
    encoder, info = AutoModel.from_pretrained(str(paths[0]), local_files_only=True,
        trust_remote_code=True, output_loading_info=True)
    bank, bank_info = AutoModel.from_pretrained(str(paths[1]), local_files_only=True,
        trust_remote_code=True, output_loading_info=True)
    for details in (info, bank_info):
        if any(details.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete pretrained weights: {details}")
    positions = bank(config["position_names"]).detach()
    if positions.shape != (len(config["channels"]), 3) or not torch.isfinite(positions).all():
        raise ValueError("Missing/invalid electrode coordinates")
    model = ReveClassifier(encoder.float(), positions, mode).to(device)
    model.loading_audit = {"encoder_loading": info, "positions_loading": bank_info,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "unused_frozen_parameters": ["encoder.cls_query_token"],
        "pooling": "time_mean_per_channel_then_concatenate", "qwen_loaded": False,
        "dtype": "float32", "position_names": config["position_names"]}
    return model


def classification_metrics(labels, predictions, dataset_ids, dataset_names):
    labels, predictions, dataset_ids = (np.asarray(x, dtype=np.int64) for x in (labels, predictions, dataset_ids))
    if not len(labels) or not (len(labels) == len(predictions) == len(dataset_ids)):
        raise ValueError("Empty or unaligned predictions")
    if not (np.isin(labels, [0, 1]).all() and np.isin(predictions, [0, 1]).all()):
        raise ValueError("Only binary labels/predictions supported")

    def summarize(y, p):
        cm = np.bincount(y * 2 + p, minlength=4).reshape(2, 2)
        rows = cm.sum(1)
        if not np.all(rows > 0):
            raise ValueError("Balanced accuracy requires both classes")
        return {"n": int(cm.sum()), "correct": int(np.trace(cm)), "accuracy": float(np.trace(cm)/cm.sum()),
                "balanced_accuracy": float(np.mean(np.diag(cm)/rows)), "confusion_matrix": cm.tolist()}

    result = summarize(labels, predictions)
    result["per_dataset"] = {name: summarize(labels[dataset_ids == i], predictions[dataset_ids == i])
                             for i, name in enumerate(dataset_names) if np.any(dataset_ids == i)}
    if not np.isin(dataset_ids, np.arange(len(dataset_names))).all():
        raise ValueError("Unknown dataset index")
    result["macro_dataset_accuracy"] = float(np.mean([v["accuracy"] for v in result["per_dataset"].values()]))
    return result
