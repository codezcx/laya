"""Reload a synthetic smoke checkpoint; check output roundtrip and padding isolation."""
import argparse
import json
from pathlib import Path

import torch

from .data import MIDataset, collate
from .model import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("smoke_dir")
    args = p.parse_args()
    root = Path(args.smoke_dir)
    checkpoint = root / "checkpoint"
    cfg = json.loads((checkpoint / "config.json").read_text())
    model = load_model(cfg, stage=cfg["saved_stage"], checkpoint=checkpoint, trainable=False)
    example = torch.load(root / "roundtrip.pt", weights_only=True)
    batch = {k: v.cuda() for k, v in example["batch"].items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(**batch)["logits"].cpu()
        torch.testing.assert_close(actual, example["logits"], atol=1e-4, rtol=1e-4)
        # The shorter sample must not depend on another sample's length or padded values.
        n = int(batch["lengths"][1])
        single = model(eeg=batch["eeg"][1:2, :, :n], lengths=batch["lengths"][1:2])["logits"].cpu()
        torch.testing.assert_close(single[0], actual[1], atol=0.1, rtol=0.02)
        changed = batch["eeg"].clone()
        changed[1, :, n:] = 10000
        padded = model(eeg=changed, lengths=batch["lengths"])["logits"].cpu()
        torch.testing.assert_close(padded, actual, atol=1e-4, rtol=1e-4)
        signal_changed = model(eeg=torch.zeros_like(batch["eeg"]), lengths=batch["lengths"])["logits"].cpu()
        signal_delta = float((signal_changed-actual).abs().max())
        assert signal_delta > 0, "Signal path is disconnected"
        # Two real training examples only for interface/finite checks, not an accuracy estimate.
        ds = MIDataset(Path(cfg["manifest_dir"]) / "train.jsonl", cfg)
        real = {k: v.cuda() for k,v in collate([ds[0], ds[1]]).items()}
        real.pop("dataset_index")
        real_out = model(**real)
        assert torch.isfinite(real_out["logits"]).all()
    report = {"checkpoint_roundtrip": True, "padding_values_ignored": True,
              "batch_vs_single_max_difference": float((single[0]-actual[1]).abs().max()),
              "signal_ablation_max_logit_difference": signal_delta,
              "real_input_shape": list(real["eeg"].shape), "real_logits_finite": True,
              "vision_modules": [n for n,_ in model.named_modules() if "visual" in n],
              "note": "Interface tests only; no accuracy estimate"}
    assert not report["vision_modules"]
    (root / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
