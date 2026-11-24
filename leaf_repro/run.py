"""Durable prepare -> four-GPU smoke -> train -> held-out evaluation controller."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, config, setup_upstream, source_hashes, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", action="store_true")
    args = p.parse_args()
    cfg = config()
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if args.start:
        with (root / "controller.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "leaf_repro.run"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Controller PID {proc.pid}; {root}")
        return
    lock = (root / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    initial_hashes = source_hashes()
    write_json(root / "launch.json", {"pid": os.getpid(), "time": time.time(), "config": cfg, "source_hashes": initial_hashes})

    def run(stage, module, *extra):
        if source_hashes() != initial_hashes:
            raise RuntimeError("Source changed after launch; refusing to mix experiment semantics")
        write_json(root / "pipeline.json", {"state": "running", "stage": stage, "time": time.time(), "pid": os.getpid()})
        with (root / f"{stage}.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", module, *extra], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(root / "child.json", {"stage": stage, "pid": child.pid})
            rc = child.wait()
        if rc:
            raise RuntimeError(f"{stage} failed ({rc}); see {stage}.log")

    try:
        run("prepare", "leaf_repro.prepare")
        setup_upstream(cfg)
        from .prepare import module_name, build_manifest
        jobs = [(ds, importlib.import_module("leaf_datasets." + module_name(ds)).SUBJECTS[0]) for ds in cfg["datasets"]]
        build_manifest(cfg, jobs, "smoke_manifest.json")
        if not (root / "smoke/complete.json").exists():
            run("smoke", "leaf_repro.train", "--smoke", "--resume")
        if not (root / "train/complete.json").exists():
            run("train", "leaf_repro.train", "--resume")
        completed = json.loads((root / "train/complete.json").read_text())
        if not (root / "test_final.json").exists() or not json.loads((root / "test_final.json").read_text()).get("completed"):
            run("evaluate_final", "leaf_repro.evaluate", "--checkpoint", str(root / "train/leaf_final.ckpt"), "--output", str(root / "test_final.json"))
        if completed["best_checkpoint"]:
            if not (root / "test_selected.json").exists() or not json.loads((root / "test_selected.json").read_text()).get("completed"):
                run("evaluate_selected", "leaf_repro.evaluate", "--checkpoint", completed["best_checkpoint"], "--output", str(root / "test_selected.json"))
        write_json(root / "pipeline.json", {"state": "complete", "time": time.time(), "source_hashes": initial_hashes})
    except Exception as e:
        write_json(root / "pipeline.json", {"state": "failed", "error": repr(e), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
