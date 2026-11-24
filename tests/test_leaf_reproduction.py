import copy
import json
from pathlib import Path
import random

import h5py
import numpy as np
import pytest
import torch

from leaf_repro.common import config, sha256, setup_upstream
from leaf_repro.prepare import build_manifest, link, POOL_SIZE


def shard(root, ds, sub, labels, predefined=False):
    p = root / "shards" / ds / (sub + ".h5")
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        for group in ("train", "valid", "test") if predefined else ("all",):
            f[group + "/X"] = np.zeros((len(labels), 65, 800), np.float32)
            f[group + "/Y"] = np.asarray(labels)
    return p


def test_pooled_every_fifth_not_per_subject(tmp_path):
    ds = "MI_BCIC_IV2a"
    subjects = [f"A{i:02d}" for i in range(1, 10)]
    for s in subjects:
        shard(tmp_path, ds, s, [0, 1, 2, 3, 0, 1])
    cfg = {"prepared_root": str(tmp_path), "datasets": [ds], "known_deviations": {}}
    m = build_manifest(cfg, [(ds, s) for s in subjects])["datasets"][ds]
    actual = [i + j * 6 for j, x in enumerate(m["splits"]["valid"]) for i in x["indices"]]
    assert actual == list(range(0, 42, 5))
    assert m["counts"]["test"] == {"windows": 12, "subjects": 2}
    train_ids = {(s["subject"], i) for s in m["splits"]["train"] for i in s["indices"]}
    valid_ids = {(s["subject"], i) for s in m["splits"]["valid"] for i in s["indices"]}
    assert train_ids.isdisjoint(valid_ids)


def test_highgamma_removes_feet_before_pooled_split(tmp_path):
    ds = "MI_HighGamma"
    shard(tmp_path, ds, "S01", [0, 1, 2, 0, 1, 2, 1])
    cfg = {"prepared_root": str(tmp_path), "datasets": [ds], "known_deviations": {}}
    m = build_manifest(cfg, [(ds, "S01")])["datasets"][ds]
    assert m["splits"]["valid"][0]["indices"] == [0]
    assert m["splits"]["train"][0]["indices"] == [1, 3, 4, 6]


def test_predefined_seed_split_preserved(tmp_path):
    ds = "EMO_SEED_3_seg4"
    shard(tmp_path, ds, "1", [0, 1, 2], predefined=True)
    cfg = {"prepared_root": str(tmp_path), "datasets": [ds], "known_deviations": {}}
    m = build_manifest(cfg, [(ds, "1")])["datasets"][ds]
    assert [m["splits"][s][0]["group"] for s in ("train", "valid", "test")] == ["train", "valid", "test"]


def test_refuses_link_overwrite(tmp_path):
    a, b, dest = tmp_path / "a", tmp_path / "b", tmp_path / "link"
    a.touch(); b.touch()
    link(a, dest)
    link(a, dest)
    with pytest.raises(RuntimeError):
        link(b, dest)


def test_native_sampling_and_lazy_reader_match(tmp_path):
    from leaf_repro.data import native, InstructionDataset, ShardArray
    ds = "MI_BCIC_IV2a"
    p = shard(tmp_path, ds, "A01", [0, 1, 2, 3])
    sh = [{"path": str(p), "group": "all", "indices": [1, 3], "subject": "A01"}]
    arr = ShardArray(sh)
    assert arr.labels().tolist() == [1, 3]
    assert len(arr) == 2 and arr[1].shape == (65, 800)
    prototypes, keys, text = native.build_prototype(native.Targets, "mpnet-base")
    assert np.allclose(np.linalg.norm(prototypes, axis=1), 1)
    assert len(keys) == len(set(sum(native.Targets.values(), [])))
    dataset = native.Tuning_Dataset([(ds, 0)], {ds: arr}, {ds: ["Right"]}, keys, text, 2000, True)
    random.seed(42)
    x, instruction, y = dataset[0]
    assert x.shape == (65, 2000) and y == keys["Right"]
    assert np.isclose(np.linalg.norm(instruction), 1)


def test_checkpoint_and_official_config():
    cfg = config(); up = setup_upstream(cfg)
    assert sha256(up / cfg["pretrain_checkpoint"]) == cfg["pretrain_sha256"]
    from load_config import load_yaml
    c = load_yaml(up / "configs/LEAF_mpnet.yaml")["instruct"]
    assert c["lr_base"] == 1e-3 and c["lr_scale_slow_params"] == 0.2
    assert c["max_epochs"] == 100 and c["warmup_epochs"] == 2
    assert cfg["batch_size_per_device"] * len(cfg["devices"]) == 1024
    assert len(cfg["datasets"]) == 13
