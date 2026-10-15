"""Regression checks for subject splits, distributed balancing and cached EEG."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eeg_align.data import MIDataset, BalancedDistributedSampler, collate, preprocessing_signature
from eeg_align.prepare_multi import subject_partition
from eeg_align.train import evaluate


class MultiDatasetTest(unittest.TestCase):
    def test_macro_dataset_accuracy_is_not_pooled_accuracy(self):
        class ToyModel(torch.nn.Module):
            def forward(self, eeg, lengths, labels):
                logits = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.], [0., 2.]])
                return {"logits": logits, "loss": torch.nn.functional.cross_entropy(logits, labels)}
        batch = {"eeg": torch.zeros(5, 3, 200), "lengths": torch.full((5,), 200),
                 "labels": torch.tensor([0, 1, 0, 1, 0]), "dataset_index": torch.tensor([0, 0, 0, 1, 1])}
        metrics = evaluate(ToyModel(), [batch], torch.device("cpu"), ["A", "B"])
        self.assertAlmostEqual(metrics["accuracy"], .4, places=6)
        self.assertAlmostEqual(metrics["macro_dataset_accuracy"], 1/3, places=6)
        self.assertEqual(metrics["per_dataset"]["B"]["n"], 2)

    def test_subject_split_stable_and_disjoint(self):
        subjects = {str(x) for x in range(1, 55)}
        a = subject_partition(subjects, "Lee2019_MI", 42)
        b = subject_partition(reversed(sorted(subjects)), "Lee2019_MI", 42)
        self.assertEqual(a, b)
        self.assertEqual(Counter(a.values()), {"train": 44, "validation": 5, "test": 5})
        self.assertEqual(set(a), subjects)

    def test_balancing_and_sharding(self):
        rows = [{"dataset": name, "label": label} for name, n in (("small", 1000), ("large", 9000))
                for label in (0, 1) for _ in range(n)]
        global_sampler = BalancedDistributedSampler(rows)
        draws = list(global_sampler)
        shards = [list(BalancedDistributedSampler(rows, r, 4)) for r in range(4)]
        self.assertEqual(draws, [i for group in zip(*shards) for i in group])
        counts = Counter((rows[i]["dataset"], rows[i]["label"]) for i in draws)
        for count in counts.values():
            self.assertLess(abs(count/len(draws)-0.25), 0.02)
        global_sampler.set_epoch(1)
        self.assertNotEqual(draws, list(global_sampler))

    def test_cache_config_and_lengths(self):
        cfg = {"channels": ["C3", "CZ", "C4"], "sample_rate": 200,
               "normalization": "per_channel_zscore_clip15", "labels": ["left_hand", "right_hand"],
               "tail_padding": "replicate_to_full_patch"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.arange(2*3*800, dtype=np.float32).reshape(2, 3, 800)
            np.save(root/"cache.npy", array)
            rows = [{"path": f"source_{i}", "label": i, "dataset_index": i,
                     "cache_file": str(root/"cache.npy"), "cache_index": i, "length": n,
                     "cache_signature": preprocessing_signature(cfg)} for i,n in enumerate([600, 800])]
            manifest = root/"data.jsonl"
            manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
            ds = MIDataset(manifest, cfg)
            batch = collate([ds[0], ds[1]])
            self.assertEqual(batch["lengths"].tolist(), [600, 800])
            self.assertEqual(batch["dataset_index"].tolist(), [0, 1])
            torch.testing.assert_close(batch["eeg"][0, :, :600], torch.from_numpy(array[0, :, :600]))
            self.assertTrue((batch["eeg"][0, :, 600:] == 0).all())
            with self.assertRaises(ValueError):
                MIDataset(manifest, {**cfg, "channels": ["C4", "CZ", "C3"]})


if __name__ == "__main__":
    unittest.main()
