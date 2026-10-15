"""Run the predefined four-GPU experiment, select using source data, then test once."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("artifacts/mi_zero_shot_v1")
PY = "laya/.venv-eeg/bin/python"
ARMS = ["legacy_ce", "harmonized_ce", "grounded_qa", "reve_mlp"]


def dump(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2)+"\n")


def launch(gpu, name, args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    log = open(ROOT/f"{name}.log", "w")
    command = [PY, "-u", "-m", "eeg_align.dg_pilot", *args]
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    log.close()
    entry = {"gpu": gpu, "name": name, "pid": process.pid, "command": command, "started": time.time()}
    with open(ROOT/"jobs.jsonl", "a") as f:
        f.write(json.dumps(entry)+"\n")
    return process, entry


def wait(jobs):
    while any(p.poll() is None for p, _ in jobs):
        time.sleep(10)
    failed = [{**e, "returncode": p.returncode} for p,e in jobs if p.returncode]
    if failed:
        dump("failed_jobs.json", failed)
        raise RuntimeError(failed)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    started = time.time()
    dump("status.json", {"phase": "waiting_for_source_cache", "started": started})
    deadline = time.time()+1800
    while len(list((ROOT/"source_cache").glob("shard_*.done.json"))) != 4:
        if time.time() > deadline:
            raise TimeoutError("Source cache not complete")
        time.sleep(10)
    files = [Path("eeg_align/dg_pilot.py"), Path(__file__), ROOT/"PROTOCOL.md",
             Path("configs/reve_qwen_mi_multi.json")]
    files.extend(Path("artifacts/reve_mi_multi/manifests_v2").glob("*.jsonl"))
    dump("input_hashes.json", {str(p): digest(p) for p in files})
    dump("status.json", {"phase": "four_gpu_pilots", "started": started})
    wait([launch(i, f"pilot_{arm}", ["train", "--arm", arm, "--steps", "400"]) for i,arm in enumerate(ARMS)])
    # The source-only amendment adds a matched preprocessing control.
    extra = "reve_mlp_legacy"
    if not (ROOT/"pilot"/extra/"complete.json").exists():
        wait([launch(3, f"pilot_{extra}", ["train", "--arm", extra, "--steps", "400"])])
    all_arms = ARMS + [extra]
    results = {arm: json.loads((ROOT/"pilot"/arm/"best_metrics.json").read_text()) for arm in all_arms}
    selected = max(ARMS[:3], key=lambda arm: results[arm]["subject_macro_accuracy"])
    selection = {"selected_qwen": selected, "frozen_arms": all_arms,
                 "refit_arms": [selected, "reve_mlp", extra], "selection_source": "Lee2019_MI 10 held-out subjects",
                 "pilot_metrics": results, "selected_at": time.time(), "target_not_evaluated": True}
    dump("selection.json", selection)
    dump("status.json", {"phase": "refit_on_three_sources", "selected": selected, "started": started})
    wait([launch(gpu, f"refit_{arm}", ["train", "--arm", arm, "--steps", str(results[arm]["step"]), "--refit"])
          for gpu,arm in zip([0,2,3], selection["refit_arms"]) if not (ROOT/"refit"/arm/"complete.json").exists()])
    checks = list((ROOT/"pilot").glob("*/best/**/*.safetensors")) + list((ROOT/"pilot").glob("*/best.pt"))
    checks += list((ROOT/"refit").glob("*/best/**/*.safetensors")) + list((ROOT/"refit").glob("*/best.pt"))
    dump("frozen_checkpoints.json", {"time": time.time(), "sha256": {str(p): digest(p) for p in checks}})
    dump("status.json", {"phase": "target_features_after_freeze", "started": started})
    wait([launch(i, f"target_cache_gpu{i}", ["cache", "--target", "--shard", str(i)]) for i in range(4)])
    dump("status.json", {"phase": "frozen_target_evaluation", "started": started})
    wait([launch(i, f"eval_pilot_{arm}", ["test", "--arm", arm]) for i,arm in enumerate(ARMS)])
    wait([launch(3, f"eval_pilot_{extra}", ["test", "--arm", extra])])
    wait([launch(i, f"eval_refit_{arm}", ["test", "--arm", arm, "--refit"])
          for i,arm in enumerate(selection["refit_arms"])])
    dump("status.json", {"phase": "complete", "started": started, "finished": time.time()})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        dump("status.json", {"phase": "failed", "error": repr(e), "time": time.time()})
        raise
