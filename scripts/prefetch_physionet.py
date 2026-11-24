"""Concurrent prefetch for the live LEAF run, without editing its guarded source.

Run: laya/.venv-eeg/bin/python scripts/prefetch_physionet.py --workers 24 --start
Uses separate resumable aria2 temporary files and publishes complete EDFs only.
Checks HTTP length and EDF structure, NOT an upstream cryptographic checksum.
Existing EDFs and the original downloader's .download files are never changed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leaf_repro.common import config, source_hashes, write_json


def edf_size(path):
    """Return the exact byte count implied by a fixed-record, 16-bit EDF header."""
    with Path(path).open("rb") as f:
        header = f.read(256)
        if len(header) != 256 or header[:8].strip() != b"0":
            raise ValueError("Not an EDF header")
        header_bytes = int(header[184:192])
        records = int(header[236:244])
        channels = int(header[252:256])
        if not 1 <= channels <= 1024 or records <= 0 or header_bytes != 256 * (1 + channels):
            raise ValueError("Unsupported EDF dimensions")
        f.seek(256 + 216 * channels)
        samples = f.read(8 * channels)
        counts = [int(samples[i:i + 8]) for i in range(0, 8 * channels, 8)]
        if len(counts) != channels or min(counts) <= 0:
            raise ValueError("Invalid EDF sample counts")
        return header_bytes + records * sum(counts) * 2


def publish(staging, target):
    """Atomically create a missing target; never replace a concurrent writer's file."""
    staging, target = Path(staging), Path(target)
    expected = edf_size(staging)
    if staging.stat().st_size != expected:
        raise ValueError("Incomplete EDF payload")
    h = hashlib.sha256()
    with staging.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    try:
        os.link(staging, target)
        created = True
    except FileExistsError:
        created = False
    # Only remove our validated staging copy; original data/partials stay intact.
    staging.unlink()
    return {"path": str(target), "bytes": expected, "sha256_downloaded": h.hexdigest(),
            "created": created, "time": time.time()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=24, choices=range(1, 33))
    p.add_argument("--start", action="store_true")
    args = p.parse_args()
    cfg = config()
    output = Path(cfg["output_root"]) / "downloads/physionet_prefetch"
    output.mkdir(parents=True, exist_ok=True)
    if args.start:
        with (output / "controller.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__, "--workers", str(args.workers)],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": proc.pid, "workers": args.workers, "status": str(output / "status.json")}))
        return
    lock = (output / "prefetch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not shutil.which("aria2c"):
        raise RuntimeError("aria2c is required")
    data = Path(cfg["prepared_root"]) / "raw/MI_PhysioNet"
    if not data.is_dir():
        raise RuntimeError("Run LEAF preparation first to resolve the raw-data path")
    if shutil.disk_usage(data).free < 125 * 1024**3:
        raise RuntimeError("Preserve 120 GiB for preprocessing plus 5 GiB download headroom")
    hashes = source_hashes()
    targets = [data / f"S{s:03d}" / f"S{s:03d}R{r:02d}.edf"
               for s in range(1, 110) for r in range(3, 15)]
    missing = [x for x in targets if not x.exists()]
    # Prefer near-term subjects; avoid duplicating currently active transfers first.
    missing.sort(key=lambda x: (x.with_suffix(".edf.download").exists(), str(x)))
    pending = {x: x.with_suffix(".edf.prefetch") for x in missing}
    lines = []
    for dest, staging in pending.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines.extend([f"https://physionet.org/files/eegmmidb/1.0.0/{dest.parent.name}/{dest.name}",
                      f"  dir={dest.parent}", f"  out={staging.name}"])
    requests = output / "requests.txt"
    requests.write_text("\n".join(lines) + "\n")
    cmd = ["aria2c", f"--input-file={requests}", f"--max-concurrent-downloads={args.workers}",
           "--max-connection-per-server=1", "--split=1", "--continue=true", "--file-allocation=none",
           "--auto-file-renaming=false", "--allow-overwrite=false", "--max-tries=12",
           "--retry-wait=3", "--connect-timeout=15", "--timeout=30", "--lowest-speed-limit=1024",
           "--summary-interval=15", "--console-log-level=warn", "--download-result=hide",
           "--enable-color=false", "--check-certificate=true"]
    started = time.time()
    write_json(output / "launch.json", {"pid": os.getpid(), "time": started, "workers": args.workers,
               "missing_at_launch": len(missing), "required": len(targets), "command": cmd,
               "source_hashes": hashes, "verification": "HTTP length + fixed-record EDF structure; no source checksum"})
    records, errors, downloaded = [], {}, 0
    with (output / "aria2.log").open("a") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL) if missing else None
        while True:
            rc = child.poll() if child else 0
            for dest, staging in list(pending.items()):
                if staging.exists() and not Path(str(staging) + ".aria2").exists():
                    try:
                        record = publish(staging, dest)
                    except (OSError, ValueError) as exc:
                        errors[str(dest)] = str(exc)
                    else:
                        records.append(record)
                        errors.pop(str(dest), None)
                        downloaded += record["bytes"]
                        del pending[dest]
                elif dest.exists():
                    # The original preparer may have completed this file first.
                    del pending[dest]
            existing = sum(x.exists() for x in targets)
            elapsed = time.time() - started
            state = {"state": "running" if rc is None else ("complete" if existing == len(targets) else "incomplete"),
                     "pid": os.getpid(), "aria2_pid": child.pid if child else None, "workers": args.workers,
                     "existing": existing, "required": len(targets), "remaining": len(targets) - existing,
                     "prefetch_published": len(records), "downloaded_bytes": downloaded,
                     "average_completed_MiB_s": downloaded / max(elapsed, 1) / 1024**2,
                     "elapsed_s": elapsed, "errors": errors, "aria2_returncode": rc, "time": time.time()}
            write_json(output / "status.json", state)
            write_json(output / "completed.json", records)
            print(json.dumps(state), flush=True)
            if rc is not None:
                break
            time.sleep(5)
    if source_hashes() != hashes:
        raise RuntimeError("Live LEAF sources changed during prefetch; investigate separately")
    if state["remaining"]:
        raise SystemExit("Some files remain; rerun this command to resume the preserved partial files")


if __name__ == "__main__":
    main()
