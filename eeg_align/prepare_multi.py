"""Audit/cache selected MI corpora; split whole subjects before mixed-data training."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import preprocessing_signature, read_trial


def subject_partition(subjects, dataset, seed):
    # Dataset-specific stable RNG; all sessions/runs of a subject stay together.
    ordered = sorted(subjects, key=lambda x: int(x))
    salt = int(hashlib.sha256(dataset.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + salt)
    rng.shuffle(ordered)
    n_hold = max(1, round(len(ordered)*0.1))
    if len(ordered) < 5:
        raise ValueError("Too few subjects")
    return {s: "test" if i < n_hold else "validation" if i < 2*n_hold else "train"
            for i, s in enumerate(ordered)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reve_qwen_mi_multi.json")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["manifest_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        raise FileExistsError("Use a new manifest directory for a new data audit")
    cache_path = out / "eeg_normalized.npy"
    jobs, assignments = [], {}
    names = cfg["dataset_names"]
    for spec in cfg["datasets"]:
        name, root = spec["name"], Path(spec["root"])
        files = sorted(p for p in root.glob("subject_*/session_*/*.pkl")
                       if any(p.name.endswith(f"label-{label}.pkl") for label in cfg["labels"]))
        if not files:
            raise ValueError(f"No MI files in {root}")
        if spec["role"] == "train":
            assignments[name] = subject_partition({p.parent.parent.name.removeprefix("subject_") for p in files}, name, cfg["seed"])
        for path in files:
            subject = path.parent.parent.name.removeprefix("subject_")
            if subject in cfg.get("excluded_subjects", {}).get(name, []):
                continue
            if spec["role"] == "train":
                split = assignments[name][subject]
            elif path.parent.name in spec["sessions"]:
                split = spec["split"]
            else:
                continue
            jobs.append((name, path, subject, split))
    # Dataset-specific windows stay intact; no assumption that all epoch starts are identical.
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype="float32",
                                    shape=(len(jobs), len(cfg["channels"]), cfg["cache_max_samples"]))
    cache[:] = 0
    rows = defaultdict(list)
    stats = {n: {"source_resolution": Counter(), "lengths": Counter(), "window_source": Counter(),
                 "units": Counter(), "subjects": set()} for n in names}
    seen_hashes, seen_events, issues = {}, set(), []
    signature = preprocessing_signature(cfg)

    def inspect(job):
        name, path, subject, split = job
        try:
            eeg, label, item = read_trial(path, cfg)
            if str(item["subject"]) != subject or path.parent.name != "session_"+str(item["session"]):
                raise ValueError("Directory/metadata identity mismatch")
            if item.get("dataset") != name:
                raise ValueError("Dataset identity mismatch")
            if eeg.shape[1] > cfg["cache_max_samples"]:
                raise ValueError("Window exceeds cache limit; do not silently truncate")
            if item.get("unit") not in ("uV", "µV"):
                raise ValueError(f"Unexpected signal unit: {item.get('unit')}")
            array = eeg.numpy()
            fingerprint = hashlib.sha256(array.tobytes()).hexdigest()
            return job, array, label, item, fingerprint, None
        except Exception as e:
            return job, None, None, None, None, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (job, array, label, item, fingerprint, error) in enumerate(pool.map(inspect, jobs)):
            name, path, subject, split = job
            if error:
                issues.append({"path": str(path), "reason": error})
                continue
            event = (name, subject, str(item["session"]), str(item["run"]), str(item["event_index"]))
            if event in seen_events or fingerprint in seen_hashes:
                issues.append({"path": str(path), "reason": "duplicate event or normalized signal",
                               "previous": seen_hashes.get(fingerprint)})
                continue
            seen_events.add(event)
            seen_hashes[fingerprint] = str(path)
            cache[i, :, :array.shape[1]] = array
            rows[split].append({"path": str(path), "label": label, "subject": subject,
                "session": item["session"], "dataset": name, "dataset_index": names.index(name),
                "length": array.shape[1], "cache_file": str(cache_path), "cache_index": i,
                "cache_signature": signature, "signal_sha256": fingerprint,
                "source_resolution": item.get("source_resolution"), "raw_name": item.get("raw_name")})
            s = stats[name]
            s["source_resolution"][str(item.get("source_resolution"))] += 1
            s["lengths"][array.shape[1]] += 1
            s["window_source"][str(item.get("window_source"))] += 1
            s["units"][str(item.get("unit"))] += 1
            s["subjects"].add(subject)
            if (i+1) % 2000 == 0:
                print(f"Audited {i+1}/{len(jobs)}", flush=True)
    cache.flush()
    for split, records in rows.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (out / "subject_split.json").write_text(json.dumps(assignments, indent=2)+"\n")
    (out / "issues.json").write_text(json.dumps(issues, indent=2)+"\n")
    for s in stats.values():
        s["subjects"] = sorted(s["subjects"], key=int)
    report = {"protocol": "Subject-disjoint within each training corpus; independent external datasets",
              "provenance_verified": False, "ready_for_exploratory_training": not issues,
              "provenance_note": "Complete local pickle checks, not raw-recording reconstruction. Inferred sources/directory fallbacks remain; no claims of unseen REVE pretraining data.",
              "normalization": cfg["normalization"], "window_policy": "retain native 3/4/5-second task windows",
              "excluded_subjects": cfg.get("excluded_subjects", {}),
              "datasets": stats, "issues_count": len(issues), "splits": {}}
    for split, records in rows.items():
        report["splits"][split] = {"count": len(records), "datasets": {name: {
            "count": sum(r["dataset"] == name for r in records),
            "subjects": len({r["subject"] for r in records if r["dataset"] == name}),
            "labels": dict(Counter(r["label"] for r in records if r["dataset"] == name))}
            for name in names if any(r["dataset"] == name for r in records)}}
    (out / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"splits": report["splits"], "issues_count": len(issues)}, indent=2), flush=True)
    if issues:
        raise RuntimeError("Audit found data issues. Review issues.json before training")


if __name__ == "__main__":
    main()
