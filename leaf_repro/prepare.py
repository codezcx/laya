"""Reuse pinned author preprocessors; remap paths without changing upstream files.

Subject shards preserve original array ordering and make preprocessing resumable.
MOABB adaptations are explicit, never silently represented as author raw exports.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import time

import h5py
import numpy as np

from .common import config, setup_upstream, source_hashes, write_json

POOL_SIZE = {"MI_BCIC_IV2a": 7, "MI_OpenBMI": 42, "MI_ShanghaiU": 20,
             "MI_HighGamma": 10, "MI_Cho2017": 40, "MI_Shin2017A": 22,
             "MI_PhysioNet": 80, "EMO_FACED": 100, "SSVEP_OpenBMI": 42, "Workload": 32}
MOABB = {"MI_Cho2017": "Cho2017", "MI_HighGamma": "Schirrmeister2017", "SSVEP_OpenBMI": "Lee2019_SSVEP"}


def module_name(ds):
    return ds.removesuffix("_seg4")


def link(target, dest):
    target, dest = Path(target).resolve(), Path(dest)
    if not target.exists():
        raise FileNotFoundError(target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.resolve() != target:
            raise RuntimeError(f"Refusing to replace existing path {dest}")
    else:
        dest.symlink_to(target, target_is_directory=target.is_dir())


def make_layout(cfg):
    src, dst = Path(cfg["raw_root"]), Path(cfg["prepared_root"]) / "raw"
    link(src / "MNE-bnci-data/~bci/database/001-2014", dst / "MI_BCIC_IV2a_mat")
    link(src / "eegbci/MNE-eegbci-data/files/eegmmidb/1.0.0", dst / "MI_PhysioNet")
    for sub in range(1, 30):
        if sub != 5:
            link(src / f"MNE-eegfnirs-data/EEG/subject {sub:02d}/with occular artifact", dst / f"MI_Shin2017A/subject {sub:02d}/with occular artifact")
    link(src / "EEGMat", dst / "Workload")
    link(src / "Processed_data", dst / "EMO_FACED/Processed_data")
    link(src / "SEED/SEED/SEED_EEG/Preprocessed_EEG", dst / "EMO_SEED_3/eeg_raw_data")
    link(src / "SEED_IV/eeg_raw_data", dst / "EMO_SEED_4/eeg_raw_data")
    link(src / "SEED-VII", dst / "EMO_SEED_7")
    base = src / "MNE-lee2019-mi-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
    for sub in range(1, 55):
        for ses in (1, 2):
            link(base / f"session{ses}/s{sub}/sess{ses:02d}_subj{sub:02d}_EEG_MI.mat",
                 dst / f"MI_OpenBMI/DB_mat/session{ses}/s{sub:02d}/EEG_MI.mat")
    for sub in range(1, 26):
        for ses in range(1, 6):
            matches = list((src / "shumi").rglob(f"sub-{sub:03d}_ses-{ses:02d}_task_motorimagery_eeg.mat"))
            if len(matches) != 1:
                raise RuntimeError(f"SHU subject {sub} session {ses}: {len(matches)} matches")
            link(matches[0], dst / f"MI_ShanghaiU/S{sub:02d}/d{ses}.mat")
    return dst


def cached_subject(ds, sub, mod, cfg):
    """Channel order comes from metadata keys matched to author's channel names."""
    import mne
    num = int(sub[1:])
    paths = list((Path(cfg["moabb_root"]) / MOABB[ds] / f"subject_{num}").rglob("*.pkl"))
    def order(p):
        return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(p)))
    paths.sort(key=order)
    mapping = ({"left_hand": 0, "right_hand": 1, "feet": 2} if ds != "SSVEP_OpenBMI"
               else {"12.0": 0, "12": 0, "8.57": 1, "8.6": 1, "6.67": 2, "6.6": 2, "5.45": 3, "5.4": 3})
    channels = mod.RAW_CH_NAMES if ds == "MI_HighGamma" else mod.CH_NAMES
    xx, yy, rows = [], [], []
    for p in paths:
        with p.open("rb") as f:
            obj = pickle.load(f)
        label = str(obj["label"])
        if label not in mapping:
            if ds == "MI_HighGamma" and label == "rest":
                continue
            raise ValueError((ds, label))
        assert obj["freq"] == 200 and obj["unit"] == "uV"
        values = {k.casefold(): v for k, v in obj["eeg"].items()}
        xx.append(np.stack([values[ch.casefold()] for ch in channels]))
        yy.append(mapping[label])
        rows.append({"path": str(p), "session": obj["session"], "run": obj["run"], "trial": obj["trial"], "label": label})
    x = np.stack(xx)
    if ds == "MI_Cho2017":
        x = x / 100  # author scale, cancels in subject median/IQR normalization
    epo = mne.EpochsArray(x, mne.create_info(channels, 200, "eeg"), verbose=False)
    if ds == "MI_HighGamma":
        epo.drop_channels(["M1", "M2"])
    epo.filter(0.3, 40, verbose=False)
    x = epo.get_data().astype(np.float32)
    if ds == "MI_Cho2017":
        x = np.pad(x, ((0, 0), (0, 0), (0, 200)), mode="edge")
    if x.shape[-1] != 800:
        raise ValueError((ds, sub, x.shape))
    x = mod.pipeline(x, mod.CH_NAMES)
    return (sub, x, np.asarray(yy, dtype=np.int64)), rows


def preprocess_one(job):
    ds, sub = job
    cfg = config()
    setup_upstream(cfg)
    import torch
    import mne
    torch.set_num_threads(1)
    mne.set_log_level("ERROR")
    mod = importlib.import_module("leaf_datasets." + module_name(ds))
    mod.RAW_DATA_FOLDER = Path(cfg["prepared_root"]) / "raw"
    folder = Path(cfg["prepared_root"]) / "shards" / ds
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sub}.h5"
    done = folder / f"{sub}.json"
    if path.exists() and done.exists():
        return json.loads(done.read_text())
    started = time.time()
    if ds in MOABB:
        result, rows = cached_subject(ds, sub, mod, cfg)
        write_json(folder / f"{sub}.sources.json", rows)
    elif ds == "MI_OpenBMI":
        result = mod.proc_one(sub, task="MI")
    elif ds == "MI_PhysioNet":
        from .downloads import ensure_physionet
        ensure_physionet(sub, mod.RAW_DATA_FOLDER / ds)
        result = mod.proc_one(sub)
    else:
        result = mod.proc_one(sub)
    sets = {"all": result[1:]} if len(result) == 3 else dict(zip(("train", "valid", "test"), (result[1:3], result[3:5], result[5:7])))
    summary = {"dataset": ds, "subject": sub, "path": str(path), "groups": {}, "seconds": None}
    tmp = path.with_suffix(".h5.partial")
    with h5py.File(tmp, "w") as f:
        f.attrs["upstream_commit"] = cfg["upstream_commit"]
        for split, (x, y) in sets.items():
            # Match the final clipping applied by the author's SEED exporters.
            if ds.startswith("EMO_SEED"):
                x = np.clip(x, -10, 10)
            x, y = x.astype(np.float32), np.asarray(y, dtype=np.int64)
            assert x.shape == (len(y), 65, 2000 if ds == "EMO_FACED" else 800), (ds, sub, x.shape, y.shape)
            assert np.isfinite(x).all() and len(y) > 0
            f.create_dataset(f"{split}/X", data=x)
            f.create_dataset(f"{split}/Y", data=y)
            summary["groups"][split] = {"n": len(y), "shape": list(x.shape), "labels": np.bincount(y).tolist(), "min": float(x.min()), "max": float(x.max())}
    tmp.replace(path)
    summary["seconds"] = time.time() - started
    write_json(done, summary)
    return summary


def build_manifest(cfg, jobs, filename="manifest.json"):
    """Indices exactly follow author's ordered-subject, pooled-every-fifth split."""
    manifest = {"schema": "leaf13-native-splits-1", "datasets": {}, "known_deviations": cfg["known_deviations"]}
    for ds in cfg["datasets"]:
        subjects = [sub for d, sub in jobs if d == ds]
        data = {"subjects": subjects, "splits": {k: [] for k in ("train", "valid", "test")}}
        offset = 0
        for si, sub in enumerate(subjects):
            path = Path(cfg["prepared_root"]) / "shards" / ds / f"{sub}.h5"
            with h5py.File(path, "r") as f:
                if ds not in POOL_SIZE:
                    for split in data["splits"]:
                        n = len(f[f"{split}/Y"])
                        data["splits"][split].append({"subject": sub, "path": str(path), "group": split, "indices": list(range(n))})
                else:
                    y = f["all/Y"][:]
                    ids = np.flatnonzero(y != 2) if ds == "MI_HighGamma" else np.arange(len(y))
                    if si < POOL_SIZE[ds]:
                        valid = (np.arange(len(ids)) + offset) % 5 == 0
                        for split, mask in (("train", ~valid), ("valid", valid)):
                            data["splits"][split].append({"subject": sub, "path": str(path), "group": "all", "indices": ids[mask].tolist()})
                        offset += len(ids)
                    else:
                        data["splits"]["test"].append({"subject": sub, "path": str(path), "group": "all", "indices": ids.tolist()})
        data["counts"] = {s: {"windows": sum(len(x["indices"]) for x in shards), "subjects": len({x["subject"] for x in shards})} for s, shards in data["splits"].items()}
        manifest["datasets"][ds] = data
    write_json(Path(cfg["prepared_root"]) / filename, manifest)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one-per-dataset", action="store_true")
    args = p.parse_args()
    cfg = config()
    setup_upstream(cfg)
    make_layout(cfg)
    jobs = []
    for ds in cfg["datasets"]:
        mod = importlib.import_module("leaf_datasets." + module_name(ds))
        jobs.extend((ds, sub) for sub in mod.SUBJECTS)
    if args.one_per_dataset:
        jobs = [next(j for j in jobs if j[0] == ds) for ds in cfg["datasets"]]
    root = Path(cfg["prepared_root"])
    fingerprints = source_hashes()
    write_json(root / "prepare_started.json", {"time": time.time(), "jobs": len(jobs), "source_hashes": fingerprints})
    done, errors = [], []
    with ProcessPoolExecutor(max_workers=cfg["prepare_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(preprocess_one, j): j for j in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                res = future.result()
                done.append(job)
                print(json.dumps({"done": len(done), "total": len(jobs), "dataset": job[0], "subject": job[1], "seconds": res["seconds"]}), flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                errors.append({"job": job, "error": repr(e)})
            write_json(root / "prepare_progress.json", {"done": len(done), "total": len(jobs), "errors": errors, "time": time.time()})
    if errors:
        raise RuntimeError(f"{len(errors)} preprocessing jobs failed; see prepare_progress.json")
    if not args.one_per_dataset:
        result = build_manifest(cfg, jobs)
        write_json(root / "complete.json", {"time": time.time(), "counts": {d: v["counts"] for d, v in result["datasets"].items()}, "source_hashes": fingerprints})


if __name__ == "__main__":
    main()
