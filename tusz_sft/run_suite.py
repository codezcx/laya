"""Supervise matched waveform -> spectrogram -> fusion/head-only experiments.

Adopts already-running first-stage torchrun processes by exact config path.
Never restarts a failed job or overwrites a checkpoint automatically.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .train_decision import json_write

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/eeg_decision_experiments"


def now():
    return datetime.now(timezone.utc).isoformat()


def running(config):
    found = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time", "status"]):
        cmd = process.info["cmdline"] or []
        if "torch.distributed.run" not in cmd or "--config" not in cmd:
            continue
        index = cmd.index("--config") + 1
        if index < len(cmd) and Path(cmd[index]).resolve() == config.resolve():
            found.append(process)
    if len(found) > 1:
        raise RuntimeError(f"More than one torchrun process uses {config}")
    return found[0] if found else None


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def completed(config):
    out = Path(json.loads(config.read_text())["out"])
    result_path = out / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete":
        return None
    latest = json.loads((out / "latest.json").read_text())["checkpoint"]
    best = json.loads((out / "best.json").read_text())["checkpoint"]
    for checkpoint in (latest, best):
        if not (Path(checkpoint) / "decision_head.safetensors").is_file():
            raise RuntimeError(f"Missing saved checkpoint: {checkpoint}")
    return result


def make_later_configs():
    template = json.loads((ROOT / "configs/waveform_lr1e5.json").read_text())
    for name, lr in (("spectrum_lr1e5", 1e-5), ("spectrum_lr5e5", 5e-5)):
        config = {**template, "input_mode": "spectrum", "learning_rate": lr,
                  "train_data": str(ARTIFACTS / "spectrum/train.fit.jsonl"),
                  "validation_data": str(ARTIFACTS / "spectrum/train.validation.jsonl"),
                  "out": str(ARTIFACTS / "runs" / name)}
        json_write(ROOT / "configs" / (name + ".json"), config)
    # This choice is made only after waveform runs complete, using internal validation.
    wave = [ARTIFACTS / "runs" / name for name in ("waveform_lr1e5", "waveform_lr5e5")]
    if all(completed(ROOT / "configs" / (p.name + ".json")) for p in wave):
        chosen = max(wave, key=lambda p: json.loads((p / "best.json").read_text())["auroc"])
        lr = json.loads((chosen / "run_config.json").read_text())["learning_rate"]
        config = {**template, "input_mode": "fusion", "learning_rate": lr,
                  "train_data": str(ARTIFACTS / "spectrum/fusion/train.fit.jsonl"),
                  "validation_data": str(ARTIFACTS / "spectrum/fusion/train.validation.jsonl"),
                  "out": str(ARTIFACTS / "runs/fusion")}
        json_write(ROOT / "configs/fusion.json", config)
        json_write(ROOT / "configs/head_only.json", {
            **template, "lora_rank": 0, "input_mode": "waveform", "out": str(ARTIFACTS / "runs/head_only")})
        json_write(ARTIFACTS / "fusion_hyperparameter_selection.json", {
            "source_run": chosen.name, "learning_rate": lr,
            "criterion": "highest waveform internal-validation AUROC; ties prefer 1e-5"})
        json_write(ROOT / "configs/native_waveform.json", {
            **template, "base": str(ROOT.parent / "CerebraGloss/DataEngine/models/Qwen3-VL-8B-Instruct"),
            "input_mode": "waveform", "grad_accum": 8, "learning_rate": 1e-5,
            "out": str(ARTIFACTS / "runs/native_waveform")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adopt-waveform", action="store_true")
    p.add_argument("--adopt-running", action="store_true",
                   help="Adopt matching live jobs after checking their assigned CUDA devices")
    p.add_argument("--poll-seconds", type=int, default=20)
    a = p.parse_args()
    os.chdir(ROOT)
    state = {"started_at": now(), "pid": os.getpid(), "status": "running", "stages": []}
    state_path = ARTIFACTS / "suite_state.json"
    plan = [(("waveform_lr1e5",), ("waveform_lr5e5",)),
            (("spectrum_lr1e5",), ("spectrum_lr5e5",)),
            (("fusion",), ("head_only", "native_waveform"))]
    import fcntl
    with (ARTIFACTS / "suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage_index, queues in enumerate(plan):
                if stage_index:
                    make_later_configs()
                stage = {"names": [n for q in queues for n in q], "queues": queues,
                         "started_at": now(), "status": "running", "jobs": []}
                state["stages"].append(stage)

                def start_next(group, index):
                    # A slot may skip previously completed jobs when a supervisor is resumed.
                    while index < len(queues[group]):
                        name = queues[group][index]
                        config = ROOT / "configs" / (name + ".json")
                        result = completed(config)
                        process = running(config)
                        gpus = list(range(group * 4, group * 4 + 4))
                        gpu_string = ",".join(map(str, gpus))
                        job = {"name": name, "config": str(config), "gpus": gpus}
                        stage["jobs"].append(job)
                        if result is not None and process is None:
                            job["status"] = "already_complete"
                            index += 1
                            continue
                        child = None
                        if process is not None:
                            if not (a.adopt_running or (stage_index == 0 and a.adopt_waveform)):
                                raise RuntimeError(f"Unexpected existing process for {name}")
                            if process.environ().get("CUDA_VISIBLE_DEVICES") != gpu_string:
                                raise RuntimeError(f"Existing {name} uses unexpected GPU devices")
                            job["adopted"] = True
                        else:
                            cfg = json.loads(config.read_text())
                            for key in ("train_data", "validation_data"):
                                if not Path(cfg[key]).is_file():
                                    raise FileNotFoundError(f"Dataset is not ready: {cfg[key]}")
                            output = Path(cfg["out"])
                            if output.exists() and any(output.iterdir()):
                                raise FileExistsError(f"Refusing to overwrite incomplete run: {output}")
                            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_string,
                                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                                   "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
                            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                       "--nproc_per_node=4", "-m", "tusz_sft.train_decision", "--config", str(config)]
                            with (ARTIFACTS / "logs" / (name + ".log")).open("w") as log:
                                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                         stderr=subprocess.STDOUT, start_new_session=True)
                            process = psutil.Process(child.pid)
                        job.update(pid=process.pid, create_time=process.create_time(), status="running")
                        json_write(state_path, state)
                        return (group, index, config, process, child, job)
                    json_write(state_path, state)
                    return None

                handles = [h for group in range(len(queues)) if (h := start_next(group, 0)) is not None]
                while handles:
                    remaining = []
                    for group, index, config, process, child, job in handles:
                        is_alive = child.poll() is None if child else alive(process)
                        if is_alive:
                            remaining.append((group, index, config, process, child, job))
                            continue
                        if child is not None and child.returncode != 0:
                            raise RuntimeError(f"{job['name']} exited with {child.returncode}")
                        result = completed(config)
                        if result is None:
                            raise RuntimeError(f"{job['name']} stopped without a complete result")
                        job.update(status="complete", ended_at=now(), steps=result["step"])
                        print(f"Completed {job['name']} at {now()}", flush=True)
                        next_job = start_next(group, index + 1)
                        if next_job is not None:
                            remaining.append(next_job)
                    handles = remaining
                    state["heartbeat_at"] = now()
                    json_write(state_path, state)
                    if handles:
                        time.sleep(a.poll_seconds)
                stage.update(status="complete", ended_at=now())
                json_write(state_path, state)
            state.update(status="complete", ended_at=now())
        except Exception as exc:
            state.update(status="failed", error=str(exc), ended_at=now())
            json_write(state_path, state)
            raise
        json_write(state_path, state)


if __name__ == "__main__":
    main()
