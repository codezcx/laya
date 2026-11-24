"""Bounded, checksum-verified public downloads; never changes an active manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .common import config, write_json, sha256


def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return s


def fetch_file(url, path, expected_sha=None, expected_size=None):
    path = Path(path)
    if path.exists():
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"Existing file has wrong size; preserving it: {path}")
        if expected_sha and sha256(path) != expected_sha:
            raise ValueError(f"Existing file checksum mismatch; preserving it: {path}")
        return path.stat().st_size
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    with session() as s, s.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        h, n = hashlib.sha256(), 0
        with tmp.open("wb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
                h.update(block)
                n += len(block)
        length = response.headers.get("Content-Length")
        if length and "Content-Encoding" not in response.headers and n != int(length):
            raise ValueError(f"Incomplete download: {path}")
        if expected_size is not None and n != expected_size:
            raise ValueError(f"Size mismatch: {path}")
        if expected_sha and h.hexdigest() != expected_sha:
            raise ValueError(f"SHA256 mismatch: {path}")
        if n == 0:
            raise ValueError(f"Empty download: {path}")
    tmp.replace(path)
    return n


def ensure_physionet(sub, base):
    """Only fetch missing author-required R03–R14, not entire unrelated corpus."""
    for run in range(3, 15):
        filename = f"{sub}R{run:02d}.edf"
        dest = Path(base) / sub / filename
        if not dest.exists():
            fetch_file(f"https://physionet.org/files/eegmmidb/1.0.0/{sub}/{filename}", dest)


def osf_inventory(folder_id, target):
    files = []
    url = f"https://api.osf.io/v2/nodes/pq7vb/files/osfstorage/{folder_id}/?page[size]=100"
    while url:
        with session() as s:
            response = s.get(url, timeout=(20, 90))
            response.raise_for_status()
            obj = response.json()
        for item in obj["data"]:
            attr = item["attributes"]
            name = attr["name"]
            if PurePosixPath(name).name != name or name in (".", ".."):
                raise ValueError("Unsafe remote filename")
            dest = Path(target) / name
            if attr["kind"] == "folder":
                files.extend(osf_inventory(item["id"], dest))
            else:
                files.append({"path": str(dest), "url": item["links"]["download"] + "?version=1",
                              "size": attr["size"], "sha256": attr["extra"]["hashes"]["sha256"], "osf_id": item["id"]})
        url = obj.get("links", {}).get("next")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["3", "4"], required=True)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    cfg = config()
    if args.start:
        root = Path(cfg["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"download_track{args.track}.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "leaf_repro.downloads", "--track", args.track],
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"Track {args.track} download PID {child.pid}")
        return
    label, folder = {"3": ("Track#3 Imagined speech classification", "5e947fd8f1353503a7d55758"),
                     "4": ("Track#4 Upper-limb movements decoding in a single-arm", "5e947fe3d697350397bdc49b")}[args.track]
    target = Path(cfg["raw_root"]) / "BCIC2020_3" / label
    root = Path(cfg["output_root"]) / "downloads" / f"track{args.track}"
    write_json(root / "status.json", {"state": "listing", "time": time.time(), "target": str(target)})
    try:
        files = osf_inventory(folder, target)
        if not any(Path(x["path"]).suffix.lower() == ".mat" for x in files):
            write_json(root / "manifest.json", files)
            for x in files:
                fetch_file(x["url"], x["path"], x["sha256"], x["size"])
            write_json(root / "status.json", {"state": "source_unavailable", "reason": "Official listing contains no recordings; inspect source readme", "time": time.time()})
            return
        needed = sum(x["size"] for x in files if not Path(x["path"]).exists())
        if needed + 120 * 1024**3 > shutil.disk_usage(cfg["raw_root"]).free:
            raise RuntimeError("Insufficient disk after reserving 120 GiB for LEAF prepared data")
        write_json(root / "manifest.json", files)
        print(json.dumps({"track": args.track, "files": len(files), "bytes": needed}), flush=True)
        completed, errors = [], []
        with ThreadPoolExecutor(3) as pool:
            work = {pool.submit(fetch_file, x["url"], x["path"], x["sha256"], x["size"]): x for x in files}
            for task in as_completed(work):
                x = work[task]
                try:
                    task.result()
                    completed.append(x["path"])
                except Exception as exc:
                    errors.append({"path": x["path"], "error": repr(exc)})
                state = {"state": "downloading", "completed": len(completed), "total": len(files), "errors": errors, "time": time.time()}
                write_json(root / "status.json", state)
                print(json.dumps(state), flush=True)
        state["state"] = "complete" if not errors else "failed"
        write_json(root / "status.json", state)
        if errors:
            raise RuntimeError("Some downloads failed; completed files are retained")
    except Exception as e:
        write_json(root / "failure.json", {"error": repr(e), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
