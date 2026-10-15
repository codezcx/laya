"""Inventory TUSZ, then render a reproducible pilot or complete split."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random

from .data import (chat_record, jev_record, read_annotations, read_jsonl,
                   windows_for_record, write_jsonl)
from .render import (RECIPE, SFREQ, load_recording, read_edf_channel_names,
                     render_epoch, select_channels)


def inspect_annotation(path):
    try:
        duration, seizures = read_annotations(path)
        select_channels(read_edf_channel_names(path.with_suffix(".edf")))
        return {"duration_seconds": duration, "seizures": seizures,
                "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except ValueError as exc:
        return {"error": str(exc)}


def inventory(root, output, skip_invalid=False):
    summary = {"data_root": str(root), "splits": {}, "errors": [],
               "label_policy": RECIPE["label_policy"], "subject_overlap": {},
               "invalid_annotation_policy": "quarantine" if skip_invalid else "error",
               "required_19_channels_checked": True}
    subjects = {}
    for split in ("train", "dev", "eval"):
        annotations = sorted((root / "edf" / split).rglob("*.csv_bi"))
        edfs = set((root / "edf" / split).rglob("*.edf"))
        annotated_edfs = {p.with_suffix(".edf") for p in annotations}
        if edfs != annotated_edfs:
            raise ValueError(f"Unpaired EDF/annotations in {split}: {len(edfs ^ annotated_edfs)}")
        counts, ids, valid_ids = Counter(), set(), set()
        records = []
        with (output / f"{split}.windows.jsonl").open("w") as windows_file, ThreadPoolExecutor(max_workers=8) as executor:
            for path, inspected in zip(annotations, executor.map(inspect_annotation, annotations)):
                edf = path.with_suffix(".edf").relative_to(root)
                subject = edf.parts[2]
                ids.add(subject)
                if "error" in inspected:
                    summary["errors"].append({"file": str(path.relative_to(root)), **inspected})
                    continue
                duration, seizures = inspected["duration_seconds"], inspected["seizures"]
                record = {"split": split, "subject": subject, "edf": str(edf),
                          **inspected}
                records.append(record)
                valid_ids.add(subject)
                counts["recordings"] += 1
                counts["duration_seconds"] += duration
                counts["seizure_seconds"] += sum(b - a for a, b in seizures)
                counts["discarded_tail_seconds"] += duration % 10
                for window in windows_for_record(record):
                    windows_file.write(json.dumps(window) + "\n")
                    counts[window["label"] + "_windows"] += 1
                    counts["boundary_windows"] += int(window["boundary"])
        write_jsonl(output / f"{split}.recordings.jsonl", records)
        subjects[split] = ids
        summary["splits"][split] = {**dict(counts), "subjects": len(ids),
                                    "included_subjects": len(valid_ids), "edf_files": len(edfs),
                                    "excluded_recordings": len(edfs) - len(records)}
    for a, b in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        overlap = sorted(subjects[a] & subjects[b])
        summary["subject_overlap"][f"{a}/{b}"] = overlap
        if overlap:
            raise ValueError(f"Subject leakage between {a}, {b}")
    (output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["errors"] and not skip_invalid:
        raise ValueError(f'{len(summary["errors"])} ineligible recordings; see inventory.json. No images built.')
    return summary


def select_windows(path, limit, seed, exclude_boundaries):
    if limit is None:
        return [w for w in read_jsonl(path) if not (exclude_boundaries and w["boundary"])]
    rng = random.Random(seed)
    reservoirs = {"seiz": [], "bckg": []}
    seen = Counter()
    for window in read_jsonl(path):
        if exclude_boundaries and window["boundary"]:
            continue
        label = window["label"]
        seen[label] += 1
        chosen = reservoirs[label]
        if len(chosen) < limit:
            chosen.append(window)
        else:
            index = rng.randrange(seen[label])
            if index < limit:
                chosen[index] = window
    if any(len(r) < limit for r in reservoirs.values()):
        raise ValueError(f"Too few windows for balanced pilot: {seen}")
    return reservoirs["seiz"] + reservoirs["bckg"]


def render_recording(job):
    root, output, windows, renderer = job
    if renderer == "v3":
        from .render_v3 import load_recording, render_epoch, RECIPE as recipe
    else:
        from .render import load_recording, render_epoch, RECIPE as recipe
    first = windows[0]
    data, metadata = load_recording(Path(root) / first["edf"], first["duration_seconds"])
    results = []
    for window in windows:
        image_path = Path(output) / "images" / window["split"] / (window["id"] + "." + recipe["format"])
        epoch = data[:, int(window["start_seconds"] * SFREQ):int(window["end_seconds"] * SFREQ)]
        render_epoch(epoch, image_path)
        results.append({**window, "image": str(image_path.resolve()), "recipe": recipe["version"],
                        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        "source": metadata})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--skip-invalid-recordings", "--skip-invalid-annotations", action="store_true",
                        help="Quarantine records with invalid annotations or missing channels; log every exclusion")
    parser.add_argument("--index", type=Path, help="Reuse an existing inventory directory")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "eval"], default=["train", "dev"])
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--heldout-per-class", type=int, default=16)
    parser.add_argument("--full", action="store_true", help="Render all windows; no class balancing")
    parser.add_argument("--keep-train-boundaries", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--renderer", choices=["v3", "bench"], default="v3")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.workers < 1 or min(args.train_per_class, args.heldout_per_class) < 1:
        parser.error("Workers and sample limits must be positive")
    args.out = args.out.resolve()
    args.data_root = args.data_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("Use a new empty output directory to avoid mixing preprocessing runs")
    index = args.index.resolve() if args.index else args.out
    if args.index:
        summary = json.loads((index / "inventory.json").read_text())
        if summary["data_root"] != str(args.data_root) or not summary.get("required_19_channels_checked") or (summary["errors"] and summary.get("invalid_annotation_policy") != "quarantine"):
            raise ValueError("Reused inventory does not match the root or contains errors")
    else:
        summary = inventory(args.data_root, args.out, args.skip_invalid_recordings)
    print(json.dumps(summary, indent=2), flush=True)
    if args.inventory_only:
        return
    if args.renderer == "v3":
        from .render_v3 import RECIPE as recipe
    else:
        recipe = RECIPE
    config = {"recipe": recipe, "data_root": str(args.data_root), "index": str(index),
              "splits": args.splits, "seed": args.seed, "full": args.full,
              "train_per_class": args.train_per_class, "heldout_per_class": args.heldout_per_class,
              "train_excludes_boundary_windows": not args.keep_train_boundaries,
              "purpose": "complete split" if args.full else "balanced engineering pilot; not a benchmark"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    groups = defaultdict(list)
    for i, split in enumerate(args.splits):
        limit = None if args.full else (args.train_per_class if split == "train" else args.heldout_per_class)
        selected = select_windows(index / f"{split}.windows.jsonl", limit, args.seed + i,
                                  split == "train" and not args.keep_train_boundaries)
        for window in selected:
            groups[window["edf"]].append(window)
    jobs = [(str(args.data_root), str(args.out), ws, args.renderer) for _, ws in sorted(groups.items())]
    # Native libraries obey these in fresh worker processes (also set by CLI examples).
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    rendered = []
    if args.workers == 1:
        outputs = map(render_recording, jobs)
        for result in outputs:
            rendered.extend(result)
            print(f"rendered {len(rendered)} windows", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(render_recording, jobs):
                rendered.extend(result)
                print(f"rendered {len(rendered)} windows", flush=True)
    report = {}
    for split in args.splits:
        rows = sorted((w for w in rendered if w["split"] == split), key=lambda w: w["id"])
        write_jsonl(args.out / f"{split}.manifest.jsonl", rows)
        write_jsonl(args.out / f"{split}.qwen.jsonl", (chat_record(w, w["image"]) for w in rows))
        write_jsonl(args.out / f"{split}.jev.jsonl", (jev_record(w, w["image"]) for w in rows))
        report[split] = {"windows": len(rows), "classes": dict(Counter(w["label"] for w in rows)),
                         "subjects": len({w["subject"] for w in rows}),
                         "boundary_windows": sum(w["boundary"] for w in rows)}
    (args.out / "render_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
