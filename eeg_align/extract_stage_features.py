"""Read-only hooks on the saved REVE -> Qwen model for stage diagnostics."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import MIDataset, collate
from .reve_baseline import SPLITS, audit_manifests, file_sha256
from .train_reve_only import write_json


STAGES = {"reve": 1536, "connector": 12288, "qwen_eeg": 12288, "qwen_answer": 4096}
SOURCE = Path("artifacts/reve_mi_multi/train_4gpu_20261015_121912")


def token_counts(lengths, patch=200, stride=180, channels=3):
    return ((lengths - patch + stride - 1) // stride + 1) * channels


def pool_tokens(tokens, counts, channels=3):
    return torch.stack([x[:int(n)].reshape(channels, int(n)//channels, -1).float().mean(1).flatten()
                        for x, n in zip(tokens, counts)])


def collect_stages(captured, counts, prefix_length, suffix_length):
    hidden = captured["hidden"]
    answer = prefix_length + counts + suffix_length - 1
    return {"reve": pool_tokens(captured["reve"], counts),
            "connector": pool_tokens(captured["connector"], counts),
            "qwen_eeg": pool_tokens(hidden[:, prefix_length:], counts),
            "qwen_answer": hidden[torch.arange(len(counts), device=counts.device), answer].float()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-local-reve-code", action="store_true")
    args = parser.parse_args()
    if not args.allow_local_reve_code:
        parser.error("Existing local REVE custom-code permission required")
    if not 0 <= args.rank < args.world:
        parser.error("Invalid rank/world")
    config = json.loads((SOURCE / "config.json").read_text())
    audit = audit_manifests(config, SOURCE / "source_and_manifest_hashes.json")
    folder = Path(args.output) / f"rank_{args.rank}"
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", {"args": vars(args), "model_config": config})
    files = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("data.py"),
             SOURCE / "lora/best/bridge.safetensors", SOURCE / "lora/best/adapter/adapter_model.safetensors"]
    write_json(folder / "provenance.json", {"files": {str(p): file_sha256(p) for p in files},
        "split_hashes": audit["hashes"], "custom_code_previously_authorized": True,
        "backbone_frozen": True, "test_features_never_used_for_selection": True})
    (folder / "status.txt").write_text("loading\n")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()
    try:
        from .model import load_model
        model = load_model(config, device=str(device), stage="lora",
                           checkpoint=SOURCE / "lora/best", trainable=False)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        captured = {}
        def connector_hook(module, inputs, output):
            captured["reve"], captured["connector"] = inputs[0], output
        def text_hook(module, inputs, output):
            captured["hidden"] = output.last_hidden_state
        hooks = [model.connector.register_forward_hook(connector_hook),
                 model.text.register_forward_hook(text_hook)]
        smoke_records = []
        for split in (SPLITS[:1] if args.smoke else SPLITS):
            (folder / "status.txt").write_text(f"extracting_{split}\n")
            ds = MIDataset(Path(config["manifest_dir"]) / f"{split}.jsonl", config)
            indices = list(range(args.rank, len(ds), args.world))
            if args.smoke:
                # One representative from each native duration; no label-based selection.
                indices = [next(i for i, r in enumerate(ds.rows) if r["length"] == n) for n in (600, 800, 1000)]
            dest = folder / split
            dest.mkdir()
            np.save(dest / "indices.npy", np.asarray(indices), allow_pickle=False)
            arrays = {name: np.lib.format.open_memmap(dest / f"{name}.npy", mode="w+", dtype="float32",
                                                      shape=(len(indices), dim)) for name, dim in STAGES.items()}
            logits = np.lib.format.open_memmap(dest / "original_logits.npy", mode="w+", dtype="float32", shape=(len(indices), 2))
            loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate)
            cursor = 0
            for step, batch in enumerate(loader):
                eeg, lengths = batch["eeg"].to(device), batch["lengths"].to(device)
                counts = token_counts(lengths)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(eeg=eeg, lengths=lengths)
                    stages = collect_stages(captured, counts, model.prefix_ids.numel(), model.suffix_ids.numel())
                with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                    torch.testing.assert_close(model.head(stages["qwen_answer"]), out["logits"], rtol=0, atol=0)
                for name, values in stages.items():
                    if not torch.isfinite(values).all():
                        raise ValueError(f"Nonfinite {name}")
                    arrays[name][cursor:cursor+len(eeg)] = values.cpu().numpy()
                logits[cursor:cursor+len(eeg)] = out["logits"].float().cpu().numpy()
                if args.smoke:
                    # Batch/solo differences from bf16 kernels are measured, not hidden.
                    for i in range(len(eeg)):
                        reference = out["logits"][i].float().cpu()
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            solo = model(eeg=eeg[i:i+1, :, :int(lengths[i])], lengths=lengths[i:i+1])["logits"][0].float().cpu()
                        smoke_records.append({"length": int(lengths[i]), "max_batch_solo_logit_delta": float((reference-solo).abs().max()),
                                              "argmax_equal": int(reference.argmax()) == int(solo.argmax())})
                cursor += len(eeg)
                captured.clear()
                if step % 25 == 0:
                    print(json.dumps({"rank": args.rank, "split": split, "done": cursor, "total": len(indices),
                                      "elapsed_s": time.monotonic()-started}), flush=True)
            for array in (*arrays.values(), logits):
                array.flush()
            assert cursor == len(indices)
        for hook in hooks:
            hook.remove()
        audit_manifests(config, SOURCE / "source_and_manifest_hashes.json")
        write_json(folder / "result.json", {"status": "complete", "seconds": time.monotonic()-started,
                    "smoke": args.smoke, "smoke_checks": smoke_records, "stages": STAGES,
                    "peak_cuda_gb": torch.cuda.max_memory_allocated()/1e9})
        (folder / "status.txt").write_text("complete\n")
    except BaseException as exc:
        (folder / "status.txt").write_text(f"failed: {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
