"""Fixed-threshold dev evaluation of an already-selected decision checkpoint."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .decision_data import DecisionCollator, file_sha256, read_manifest
from .decision_model import load_model
from .train_decision import evaluate, json_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    rank, local, world = (int(os.environ.get(k, v)) for k, v in
                           (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    try:
        if a.out.exists() and any(a.out.iterdir()):
            raise FileExistsError("Evaluation output already exists")
        cfg = json.loads((a.checkpoint / "decision_config.json").read_text())
        mode = cfg.get("input_mode", "waveform")
        rows = read_manifest(a.manifest, "dev", mode=mode, source_split="dev")
        started = time.monotonic()
        model, processor = load_model(checkpoint=a.checkpoint, device=device,
                                      trainable=False, gradient_checkpointing=False)
        collator = DecisionCollator(processor, cfg["max_pixels"], cfg["max_length"], cfg["prompt"])
        metrics, predictions = evaluate(model, rows, collator, device, rank, world, a.batch_size)
        if rank == 0:
            a.out.mkdir(parents=True)
            by_id = {row["id"]: row for row in rows}
            predictions = [{**pred, "subject": by_id[pred["id"]]["subject"],
                            "boundary": by_id[pred["id"]]["boundary"]} for pred in predictions]
            json_write(a.out / "predictions.json", predictions)
            result = {"status": "complete", **metrics, "input_mode": mode, "world_size": world,
                      "checkpoint": str(a.checkpoint.resolve()), "base": cfg["source"]["path"],
                      "manifest_sha256": file_sha256(a.manifest), "subjects": len({r["subject"] for r in rows}),
                      "elapsed_seconds_including_load": time.monotonic() - started,
                      "interpretation": "Fixed sampled official-dev windows excluded from available prior training window lists. Patient overlap remains for the EEG RL base; not evidence of unseen-patient generalization."}
            json_write(a.out / "metrics.json", result)
            print(json.dumps(result), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
