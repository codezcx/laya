from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / "configs/leaf_reproduction.json").read_text())
    for key in ("prepared_root", "output_root"):
        cfg[key] = str(ROOT / cfg[key])
    return cfg


def setup_upstream(cfg):
    upstream = Path(cfg["upstream"])
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != cfg["upstream_commit"]:
        raise RuntimeError(f"Upstream revision changed: {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "diff", "--name-only", "HEAD"], text=True).strip():
        raise RuntimeError("Upstream tracked source has modifications")
    sys.path.insert(0, str(upstream))
    return upstream


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def source_hashes():
    cfg = config()
    paths = list((ROOT / "leaf_repro").glob("*.py")) + [ROOT / "configs/leaf_reproduction.json"]
    upstream = Path(cfg["upstream"])
    paths += list(upstream.glob("*.py")) + list((upstream / "leaf_datasets").glob("*.py"))
    paths += list((upstream / "configs").glob("*.yaml"))
    paths += [upstream / "text_embeddings/mpnet-base.txt", upstream / "text_embeddings/mpnet-base.npy"]
    return {str(p): sha256(p) for p in sorted(paths)}
