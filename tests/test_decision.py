import ast
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from tusz_sft.decision_model import EEGDecisionModel, last_prompt_state
from tusz_sft.render import CHANNELS
from tusz_sft.render_v3 import render_epoch
from tusz_sft.prepare_spectrum import power_spectrogram
from tusz_sft.decision_data import RECIPES, file_sha256, read_manifest
from tusz_sft.train_decision import load_resume_checkpoint


class DecisionTests(unittest.TestCase):
    def test_resume_checks_owner_latest_identity_and_completion_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            ck = out / "checkpoints/step-000002"
            ck.mkdir(parents=True)
            identity = {"hyperparameters": {"input_mode": "waveform"}, "world_size": 4}
            torch.save({"identity": identity, "epoch": 0, "step": 2}, ck / "training_state.pt")
            (out / "latest.json").write_text(json.dumps({"checkpoint": str(ck)}))
            (out / "run_config.json").write_text("original config")
            before = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(load_resume_checkpoint(ck, out, identity, 3, 4)["step"], 2)
            for checkpoint, directory, expected, epochs, limit in (
                (ck, Path(d) / "wrong", identity, 3, 4),
                (out / "checkpoints/step-000001", out, identity, 3, 4),
                (ck, out, {**identity, "world_size": 8}, 3, 4),
                (ck, out, identity, 0, 4),
                (ck, out, identity, 3, 2),
            ):
                with self.assertRaises(ValueError):
                    load_resume_checkpoint(checkpoint, directory, expected, epochs, limit)
            after = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_dev_cannot_enter_training_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            image = d / "image.png"
            image.write_bytes(b"fixture")
            row = {"id": "window", "subject": "subject", "split": "dev", "label": "seiz",
                   "recipe": RECIPES["waveform"], "image": str(image), "image_sha256": file_sha256(image)}
            manifest = d / "dev.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                read_manifest(manifest, "train")
            with self.assertRaises(ValueError):
                read_manifest(manifest, "dev")
            self.assertEqual(read_manifest(manifest, "dev", source_split="dev"), [row])

    def test_spectrogram_frequency_and_absolute_power_scaling(self):
        t = np.arange(2000) / 200
        data = np.tile(10 * np.sin(2 * np.pi * 10 * t), (19, 1))
        frequency, time, power = power_spectrogram(data)
        self.assertEqual(frequency[power[0].mean(axis=1).argmax()], 10)
        self.assertEqual((time[0], time[-1]), (1., 9.))
        _, _, louder = power_spectrogram(data * 2)
        peak = np.flatnonzero(frequency == 10)[0]
        np.testing.assert_allclose(louder[:, peak] - power[:, peak], 20 * np.log10(2), atol=1e-6)

    def test_last_prompt_for_left_and_right_padding(self):
        states = torch.arange(24).reshape(2, 4, 3).float()
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
        torch.testing.assert_close(last_prompt_state(states, mask), torch.stack([states[0, 3], states[1, 1]]))

    def test_decision_head_equals_restricted_vocab_and_backpropagates(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10, 8)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        torch.manual_seed(1)
        base = Backbone()
        vocabulary = nn.Linear(8, 20, bias=False)
        tokens = [3, 7]
        model = EEGDecisionModel(base, vocabulary.weight[tokens].detach(), {})
        ids = torch.tensor([[2, 3, 5], [1, 5, 7]])
        mask = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=mask)
        expected = vocabulary(base(input_ids=ids).last_hidden_state[:, -1])[:, tokens]
        torch.testing.assert_close(logits, expected)
        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(base.embedding.weight.grad.abs().sum().item(), 0)
        self.assertFalse(any("lm_head" in name for name, _ in model.named_parameters()))

    def test_v3_pixel_parity_with_local_training_renderer(self):
        upstream = Path("/inspire/hdd/global_user/zhangchenxi-253108310322/CerebraGloss/DataEngine/render_images_v3.py")
        if not upstream.is_file():
            self.skipTest("Local AAAI source is not mounted")
        tree = ast.parse(upstream.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        scope = {"np": np, "plt": plt, "FS": 200, "STD": list(CHANNELS), "SCALE_UV": 100.}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(upstream), "exec"), scope)
        t = np.arange(2000) / 200
        data = np.array([i * np.sin(2 * np.pi * (i + 1) * t) for i in range(19)], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "input.npy", data)
            with plt.rc_context(plt.rcParamsDefault):
                scope["render"](d / "input.npy", d / "original.png")
            render_epoch(data, d / "ours.png")
            self.assertEqual((d / "original.png").read_bytes(), (d / "ours.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
