"""Image-only decision prompts, audited manifests, and subject-held-out validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
import torch

from .data import QUESTION, STATE, read_jsonl, write_jsonl
from .render_v3 import RECIPE

LABELS = ["bckg", "seiz"]
PROMPT = STATE + "\n" + QUESTION
PROMPTS = {
    "waveform": PROMPT,
    "spectrum": "Power spectrograms of 19 scalp EEG channels in common-average reference, "
                "covering one 10-second segment. Each panel is one electrode. The vertical axis is "
                "frequency (0.5-70 Hz), the horizontal axis is time, and color is absolute power "
                "density in dB relative to 1 microvolt squared per Hz, on a shared scale.\n" + QUESTION,
    "fusion": "Two views of the SAME 10-second, 19-channel EEG segment in common-average reference. "
              "The first image shows waveforms, with 200 microvolt channel spacing and negative voltage upward. "
              "The second shows per-channel power spectrograms (0.5-70 Hz), with time horizontally, "
              "frequency vertically, and a shared absolute dB power-density color scale.\n" + QUESTION,
}
RECIPES = {"waveform": RECIPE["version"], "spectrum": "tusz-stft-2s-025s-fixed-db-v1",
           "fusion": "tusz-waveform-and-stft-v1"}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path, split, verify_images=True, mode="waveform", source_split="train"):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate window ID")
        seen.add(row["id"])
        if row["split"] != split or row.get("source_split", row["split"]) != source_split:
            raise ValueError(f"Expected split={split}, source_split={source_split}")
        if row.get("recipe") != RECIPES[mode]:
            raise ValueError(f"Expected {mode} recipe {RECIPES[mode]}")
        if row["label"] not in LABELS or not row.get("subject"):
            raise ValueError("Invalid label/subject")
        images = row.get("images", [row["image"]])
        hashes = row.get("image_sha256s", [row["image_sha256"]])
        if len(images) != (2 if mode == "fusion" else 1) or len(images) != len(hashes):
            raise ValueError("Wrong image count for selected input mode")
        for image, digest in zip(images, hashes):
            image = Path(image)
            if not image.is_file():
                raise FileNotFoundError(image)
            if verify_images and file_sha256(image) != digest:
                raise ValueError(f"Image hash mismatch: {image}")
    return rows


def load_datasets(train_path, validation_path=None, mode="waveform"):
    train = read_manifest(train_path, "train", mode=mode)
    validation = read_manifest(validation_path, "validation", mode=mode) if validation_path else []
    if set(r["label"] for r in train) != set(LABELS):
        raise ValueError("Training requires both classes")
    if validation and set(r["label"] for r in validation) != set(LABELS):
        raise ValueError("Internal validation requires both classes")
    if {r["subject"] for r in train} & {r["subject"] for r in validation}:
        raise ValueError("Subject leakage between training and validation")
    if {r["image_sha256"] for r in train} & {r["image_sha256"] for r in validation}:
        raise ValueError("Duplicate images across training and validation")
    return train, validation


class DecisionCollator:
    def __init__(self, processor, max_pixels=1500000, max_length=4096, prompt=PROMPT):
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_length = max_length
        self.prompt = prompt
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, rows):
        texts, images = [], []
        for row in rows:
            # No answer, patient ID, filename or annotation is placed in the prompt.
            paths = row.get("images", [row["image"]])
            message = [{"role": "user", "content": [
                *[{"type": "image"} for _ in paths], {"type": "text", "text": self.prompt}]}]
            texts.append(self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True))
            for path in paths:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
        inputs = self.processor(text=texts, images=images, padding=True,
                                return_tensors="pt", images_kwargs={
                                    "min_pixels": min(65536, self.max_pixels), "max_pixels": self.max_pixels})
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError("Input exceeds max_length; reduce max_pixels, never truncate image tokens")
        labels = torch.tensor([LABELS.index(r["label"]) for r in rows], dtype=torch.long)
        return dict(inputs), labels


def main():
    parser = argparse.ArgumentParser(description="Hold out subjects from official train; never use dev/eval")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between 0 and 1")
    rows = read_manifest(args.manifest, "train")
    subjects = sorted({r["subject"] for r in rows})
    random.Random(args.seed).shuffle(subjects)
    n = max(1, round(len(subjects) * args.validation_fraction))
    held = set(subjects[:n])
    fit = [r for r in rows if r["subject"] not in held]
    val = [{**r, "split": "validation", "source_split": "train"} for r in rows if r["subject"] in held]
    if any(set(r["label"] for r in group) != set(LABELS) for group in (fit, val)):
        raise ValueError("Chosen subject split lacks a class; use more data or another fixed seed")
    out = args.manifest.parent
    paths = [out / "train.fit.jsonl", out / "train.validation.jsonl", out / "subject_split.json"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Split outputs already exist; use a new dataset directory")
    write_jsonl(paths[0], fit)
    write_jsonl(paths[1], val)
    summary = {"seed": args.seed, "validation_fraction": args.validation_fraction,
               "source_manifest_sha256": file_sha256(args.manifest),
               "train": {"windows": len(fit), "classes": dict(Counter(r["label"] for r in fit))},
               "validation": {"windows": len(val), "classes": dict(Counter(r["label"] for r in val))},
               "validation_subjects": sorted(held)}
    paths[2].write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
