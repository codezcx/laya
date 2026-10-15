"""Prepare a fixed dev subset absent from available prior training window lists.

This removes exact window overlap, NOT patient overlap or adjacent-window effects.
It cannot certify an unknown checkpoint's complete pretraining provenance.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from .data import read_jsonl, write_jsonl
from .decision_data import file_sha256
from .prepare import render_recording


def window_key(path):
    match = re.fullmatch(r"([a-z]{8}_s\d+_t\d+)_w(\d+)", Path(path).stem)
    return (match[1], int(match[2])) if match else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=512)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261014)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = root.parent / "CerebraGloss/DataEngine"
    data_root = root.parent / "mne_data/TUSZ"
    prior_files = [upstream / "verl_data/train_v3.parquet",
                   upstream / "sft_recipe/recipe_train.jsonl",
                   upstream / "sft_recipe/recipe_train_v3render.jsonl"]
    keys = set()
    for source in prior_files:
        if source.suffix == ".parquet":
            paths = [im["image"] for row in pq.read_table(source, columns=["images"]).column("images").to_pylist() for im in row]
        else:
            paths = [im for row in read_jsonl(source) for im in row.get("images", []) if isinstance(im, str)]
        keys.update(key for path in paths if (key := window_key(path)) is not None)
    subjects = {stem.split("_")[0] for stem, index in keys}
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Use an empty output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    counts = Counter()
    selected = {"bckg": [], "seiz": []}
    excluded = 0
    for row in read_jsonl(root / "artifacts/tusz_cerebragloss/index_final/dev.windows.jsonl"):
        key = (Path(row["edf"]).stem, int(row["start_seconds"] // 10))
        if key in keys:
            excluded += 1
            continue
        label = row["label"]
        counts[label] += 1
        group = selected[label]
        if len(group) < a.per_class:
            group.append(row)
        else:
            index = rng.randrange(counts[label])
            if index < a.per_class:
                group[index] = row
    if any(len(rows) != a.per_class for rows in selected.values()):
        raise ValueError(f"Insufficient candidates: {counts}")
    rows = sorted(selected["bckg"] + selected["seiz"], key=lambda r: r["id"])
    write_jsonl(a.out / "selected_windows.jsonl", rows)
    report = {"source_split": "dev", "seed": a.seed, "per_class": a.per_class,
              "excluded_prior_windows": excluded, "remaining_candidates": dict(counts),
              "selected_windows": len(rows), "subjects": len({r["subject"] for r in rows}),
              "selected_subjects_in_prior_corpora": len({r["subject"] for r in rows} & subjects),
              "boundary_windows": sum(r["boundary"] for r in rows),
              "prior_corpus_sha256": {str(p): file_sha256(p) for p in prior_files},
              "limitation": "Exact-window exclusion only; patient overlap remains. Not a clean unseen-patient test."}
    (a.out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    groups = defaultdict(list)
    for row in rows:
        groups[row["edf"]].append(row)
    rendered = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(render_recording, [(str(data_root), str(a.out), rows, "v3") for rows in groups.values()]):
            rendered.extend(result)
            print(f"rendered {len(rendered)}/{len(rows)} held-out dev windows", flush=True)
    write_jsonl(a.out / "dev.manifest.jsonl", sorted(rendered, key=lambda r: r["id"]))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
