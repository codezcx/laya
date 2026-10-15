"""Wait for the training suite, then evaluate locked selections in two GPU groups."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .run_suite import ARTIFACTS, ROOT, alive, completed, running
from .summarize_experiments import RUNS
from .train_decision import json_write


def main():
    os.chdir(ROOT)
    state = {"pid": os.getpid(), "status": "waiting_for_training", "jobs": []}
    path = ARTIFACTS / "evaluation_state.json"
    import fcntl
    with (ARTIFACTS / "evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                suite = json.loads((ARTIFACTS / "suite_state.json").read_text())
                state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                json_write(path, state)
                if suite["status"] == "failed":
                    raise RuntimeError("Training suite failed; evaluation was not started")
                if suite["status"] == "complete":
                    if not all(completed(ROOT / "configs" / (name + ".json")) for name in RUNS):
                        raise RuntimeError("Suite status disagrees with individual results")
                    if not any(running(ROOT / "configs" / (name + ".json")) for name in RUNS):
                        break
                else:
                    try:
                        supervisor = psutil.Process(suite["pid"])
                        live = alive(supervisor) and "tusz_sft.run_suite" in supervisor.cmdline()
                    except psutil.NoSuchProcess:
                        live = False
                    if not live:
                        raise RuntimeError("Training supervisor disappeared before completion")
                time.sleep(20)
            tasks = []
            for name in ["rl_initial", *RUNS]:
                if name == "rl_initial":
                    checkpoint = ARTIFACTS / "runs/waveform_lr1e5/checkpoints/step-000000"
                else:
                    checkpoint = Path(json.loads((ARTIFACTS / "runs" / name / "best.json").read_text())["checkpoint"])
                cfg = json.loads((checkpoint / "decision_config.json").read_text())
                mode = cfg.get("input_mode", "waveform")
                directory = {"waveform": "dev_waveform", "spectrum": "dev_spectrum", "fusion": "dev_spectrum/fusion"}[mode]
                manifest = ARTIFACTS / directory / "dev.manifest.jsonl"
                if not manifest.is_file():
                    raise FileNotFoundError(manifest)
                tasks.append((name, checkpoint, manifest))
            state["status"] = "evaluating"
            for start in range(0, len(tasks), 2):
                children = []
                for group, (name, checkpoint, manifest) in enumerate(tasks[start:start + 2]):
                    out = ARTIFACTS / "evaluations" / name
                    if out.exists() and any(out.iterdir()):
                        raise FileExistsError(f"Refusing to overwrite an evaluation: {out}")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(group * 4, group * 4 + 4))),
                           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
                    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                               "-m", "tusz_sft.evaluate_decision", "--checkpoint", str(checkpoint),
                               "--manifest", str(manifest), "--out", str(out)]
                    with (ARTIFACTS / "logs" / f"evaluate_{name}.log").open("w") as log:
                        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    job = {"name": name, "pid": process.pid, "checkpoint": str(checkpoint),
                           "manifest": str(manifest), "status": "running"}
                    state["jobs"].append(job)
                    children.append((process, job, out))
                while children:
                    remaining = []
                    for process, job, out in children:
                        rc = process.poll()
                        if rc is None:
                            remaining.append((process, job, out))
                        elif rc != 0 or not (out / "metrics.json").is_file():
                            raise RuntimeError(f"Evaluation failed: {job['name']} (exit {rc})")
                        else:
                            job["status"] = "complete"
                    children = remaining
                    state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                    json_write(path, state)
                    if children:
                        time.sleep(20)
            subprocess.run([sys.executable, "-m", "tusz_sft.summarize_experiments"], cwd=ROOT, check=True)
            state["status"] = "complete"
        except Exception as exc:
            state.update(status="failed", error=str(exc))
            json_write(path, state)
            raise
        json_write(path, state)


if __name__ == "__main__":
    main()
