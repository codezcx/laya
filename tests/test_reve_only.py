"""Network-free REVE-only baseline regression tests (no custom REVE code runs)."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn

from eeg_align.reve_baseline import ReveClassifier, classification_metrics, load_reve_classifier, pool_channels
from eeg_align.train_reve_only import checkpoint_state, restore_checkpoint, verify_gradients, write_json


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(embed_dim=4, patch_size=4, patch_overlap=1)
        self.projection = nn.Linear(4, 4)
        self.cls_query_token = nn.Parameter(torch.ones(1, 1, 4))

    def forward(self, eeg, pos):
        return self.projection(eeg.unfold(-1, 4, 3))


class ReveOnlyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.eeg = torch.randn(3, 3, 11)
        self.lengths = torch.tensor([5, 11, 8])
        self.labels = torch.tensor([0, 1, 0])

    def model(self, mode):
        return ReveClassifier(ToyEncoder(), torch.zeros(3, 3), mode)

    def test_time_pool_preserves_channel_identity(self):
        x = torch.zeros(1, 3, 2, 4)
        x[:, 0] = 1
        x[:, 1] = 2
        x[:, 2] = 3
        self.assertEqual(pool_channels(x).tolist(), [[1]*4 + [2]*4 + [3]*4])

    def test_padding_and_batch_composition_do_not_change_features(self):
        model = self.model("linear_probe").eval()
        features = model.encode(self.eeg, self.lengths)
        changed = self.eeg.clone()
        for i, n in enumerate(self.lengths.tolist()):
            changed[i, :, n:] = 1e6
            solo = model.encode(self.eeg[i:i+1, :, :n], torch.tensor([n]))
            torch.testing.assert_close(features[i:i+1], solo)
        torch.testing.assert_close(features, model.encode(changed, self.lengths))

    def test_linear_probe_freezes_encoder_in_eval_mode(self):
        model = self.model("linear_probe").train()
        self.assertFalse(model.encoder.training)
        before = copy.deepcopy(model.encoder.state_dict())
        opt = torch.optim.SGD(model.head.parameters(), lr=.01)
        model(eeg=self.eeg, lengths=self.lengths, labels=self.labels)["loss"].backward()
        verify_gradients(model)
        opt.step()
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(sum(p.requires_grad for p in model.parameters()), 2)

    def test_full_sft_updates_forward_encoder_parameters(self):
        model = self.model("full_sft").train()
        self.assertTrue(model.encoder.training)
        before = model.encoder.projection.weight.detach().clone()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        model(eeg=self.eeg, lengths=self.lengths, labels=self.labels)["loss"].backward()
        result = verify_gradients(model)
        self.assertEqual(result["encoder_gradient_tensors"], 2)
        opt.step()
        self.assertFalse(torch.equal(before, model.encoder.projection.weight))

    def test_cached_features_match_raw_linear_probe(self):
        model = self.model("linear_probe").eval()
        features = model.encode(self.eeg, self.lengths)
        torch.testing.assert_close(model(features=features)["logits"],
                                   model(eeg=self.eeg, lengths=self.lengths)["logits"])
        with self.assertRaises(ValueError):
            self.model("full_sft")(features=features)

    def test_save_reload_preserves_both_modes(self):
        for mode in ("linear_probe", "full_sft"):
            model = self.model(mode).eval()
            original = model(eeg=self.eeg, lengths=self.lengths)["logits"].detach()
            state = checkpoint_state(model, mode)
            with torch.no_grad():
                model.head.weight.add_(10)
                if mode == "full_sft":
                    model.encoder.projection.weight.add_(10)
            restore_checkpoint(model, state, mode)
            torch.testing.assert_close(model(eeg=self.eeg, lengths=self.lengths)["logits"], original)

    def test_metrics_macro_vs_pooled(self):
        m = classification_metrics([0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1], ["A", "B"])
        self.assertAlmostEqual(m["accuracy"], .5)
        self.assertAlmostEqual(m["macro_dataset_accuracy"], .375)
        self.assertEqual(m["confusion_matrix"], [[2, 1], [2, 1]])

    def test_custom_model_code_needs_explicit_permission(self):
        with self.assertRaises(PermissionError):
            load_reve_classifier({}, "linear_probe", "cpu")

    def test_transformers_loading_audit_sets_are_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "audit.json"
            write_json(target, {"missing_keys": set(), "nested": {"keys": {"b", "a"}}})
            self.assertEqual(json.loads(target.read_text()),
                             {"missing_keys": [], "nested": {"keys": ["a", "b"]}})


if __name__ == "__main__":
    unittest.main()
