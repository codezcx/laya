#!/usr/bin/env python3
"""Verify both 32-token resamplers; NOT a training or EEG accuracy experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.token_resampler import ResamplerConfig, TokenResampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/token_resamplers.json")
    parser.add_argument("--tokens", type=Path, help="Optional NPZ: tokens [B,N,D], valid bool [B,N]")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--length", type=int, default=1568)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.length) < 1:
        parser.error("threads, batch-size and length must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    configs = {k: ResamplerConfig.from_json(args.config, k) for k in ("cross_attention", "qformer")}
    if any(c.num_queries != 32 for c in configs.values()):
        parser.error("This check requires exactly 32 output tokens for both variants")
    if len({(c.input_dim, c.hidden_dim) for c in configs.values()}) != 1:
        parser.error("Both variants must share input and output widths")
    if args.tokens is None:
        tokens = torch.randn(args.batch_size, args.length, configs["cross_attention"].input_dim)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool)
        source = {"kind": "synthetic_random_tokens", "seed": args.seed}
    else:
        with np.load(args.tokens, allow_pickle=False) as data:
            tokens = torch.from_numpy(data["tokens"].copy())
            valid = torch.from_numpy(data["valid"].copy()) if "valid" in data else None
        source = {"kind": "provided_tokens", "path": str(args.tokens.resolve()),
                  "sha256": hashlib.sha256(args.tokens.read_bytes()).hexdigest()}
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    summary = {
        "status": "running", "trained": False, "optimizer_steps": 0,
        "scope": "Token forward/backward only; no EEG encoder or JEV inference, no accuracy",
        "input": source, "input_shape": list(tokens.shape), "seed": args.seed,
        "device": str(device), "torch_version": str(torch.__version__),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), ROOT / "eeg_align/token_resampler.py")
        },
        "variants": {},
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for kind, config in configs.items():
        torch.manual_seed(args.seed)
        model = TokenResampler(config).to(device).eval()
        x = tokens.to(device).detach().requires_grad_(True)
        mask = None if valid is None else valid.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        compressed, compressed_valid = model(x, mask)
        target = torch.randn_like(compressed)
        loss = (compressed - target).square().mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        bad_grads = [n for n, p in model.named_parameters()
                     if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.any()]
        if bad_grads or not torch.isfinite(x.grad).all() or not x.grad.any():
            raise RuntimeError(f"Gradient check failed for {kind}: {bad_grads}")
        if not torch.isfinite(compressed).all() or compressed.shape[1] != 32:
            raise RuntimeError(f"Invalid compressed output for {kind}")
        np.savez_compressed(
            args.out / f"{kind}.npz", tokens=compressed.detach().float().cpu().numpy(),
            valid=compressed_valid.cpu().numpy(),
        )
        row = {
            "config": config.to_dict(), "parameters": sum(p.numel() for p in model.parameters()),
            "output_shape": list(compressed.shape), "all_finite": True,
            "all_parameter_gradients_finite_nonzero": True,
            "input_gradient_finite_nonzero": True, "forward_backward_seconds": elapsed,
        }
        summary["variants"][kind] = row
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(kind, json.dumps(row), flush=True)
        del model, x, compressed, compressed_valid, target, loss
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Untrained token checks complete:", summary_path, flush=True)


if __name__ == "__main__":
    main()
