"""Load a saved decision adapter/head and classify images in one forward pass."""
import argparse
import json
from pathlib import Path

import torch

from .decision_data import DecisionCollator, LABELS
from .decision_model import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(2)
    model, processor = load_model(checkpoint=a.checkpoint, device=a.device, trainable=False,
                                  gradient_checkpointing=False)
    cfg = model.metadata
    collator = DecisionCollator(processor, cfg["max_pixels"], cfg["max_length"], cfg["prompt"])
    results = []
    if cfg.get("input_mode") == "fusion" and len(a.image) != 2:
        p.error("Fusion expects exactly two --image paths: waveform then spectrogram")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(a.image), a.batch_size):
            rows = [{"image": path, "label": "bckg"} for path in a.image[start:start + a.batch_size]]
            if cfg.get("input_mode") == "fusion":
                rows = [{"image": a.image[0], "images": a.image, "label": "bckg"}]
            inputs, _ = collator(rows)
            probabilities = model(**{k: v.to(a.device) for k, v in inputs.items()}).softmax(-1).cpu()
            results.extend({"image": row["image"], "label": LABELS[int(probs.argmax())],
                            "probabilities": dict(zip(LABELS, probs.tolist()))} for row, probs in zip(rows, probabilities))
            if cfg.get("input_mode") == "fusion":
                results[-1]["images"] = a.image
                break
    output = json.dumps(results, indent=2) + "\n"
    if a.out:
        a.out.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
