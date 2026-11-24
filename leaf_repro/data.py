"""Lazy subject-shard access and the author's unmodified instruction sampler."""
from collections import OrderedDict
import json
from pathlib import Path

import h5py
import numpy as np

from .common import config, setup_upstream

CFG = config()
UPSTREAM = setup_upstream(CFG)
import b_instruct_tuning as native
import init_text_embeddings

init_text_embeddings.EMBEDDING_ROOT = UPSTREAM / "text_embeddings"
native.Targets = {ds: native.Targets[ds] for ds in CFG["datasets"]}


class ShardArray:
    def __init__(self, shards):
        self.shards = shards
        self.ends = np.cumsum([len(s["indices"]) for s in shards])
        self.handles = OrderedDict()

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, i):
        pos = int(np.searchsorted(self.ends, i, side="right"))
        s = self.shards[pos]
        index = s["indices"][i - (int(self.ends[pos - 1]) if pos else 0)]
        path = s["path"]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", locking=False)
        self.handles.move_to_end(path)
        while len(self.handles) > 16:
            _, handle = self.handles.popitem(last=False)
            handle.close()
        return self.handles[path][s["group"] + "/X"][index]

    def __getstate__(self):
        return {**self.__dict__, "handles": OrderedDict()}

    def labels(self):
        chunks = []
        for s in self.shards:
            with h5py.File(s["path"], "r", locking=False) as f:
                chunks.append(f[s["group"] + "/Y"][:][s["indices"]])
        return np.concatenate(chunks)


class InstructionDataset(native.Tuning_Dataset):
    def __init__(self, manifest, split, key2idx, text2emb, max_per_dataset=None):
        key, xx, yy = [], {}, {}
        self.dataset_ids = list(native.Targets)
        for ds in self.dataset_ids:
            arr = ShardArray(manifest["datasets"][ds]["splits"][split])
            xx[ds] = arr
            labels = arr.labels()
            assert labels.min() >= 0 and labels.max() < len(native.Targets[ds]), (ds, np.unique(labels))
            yy[ds] = [native.Targets[ds][int(y)] for y in labels]
            size = len(arr) if max_per_dataset is None else min(len(arr), max_per_dataset)
            key.extend((ds, i) for i in range(size))
        super().__init__(key, xx, yy, key2idx, text2emb, 2000, training=split == "train")

    def __getitem__(self, i):
        return (*super().__getitem__(i), self.dataset_ids.index(self.Key[i][0]))


def read_manifest(smoke=False):
    name = "smoke_manifest.json" if smoke else "manifest.json"
    return json.loads((Path(CFG["prepared_root"]) / name).read_text())
