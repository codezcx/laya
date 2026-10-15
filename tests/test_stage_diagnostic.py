import unittest
import numpy as np
import torch

from eeg_align.extract_stage_features import collect_stages, pool_tokens, token_counts
from eeg_align.fit_stage_probe import fit_logistic, group_weights, metrics_from_scores


class StageDiagnosticTest(unittest.TestCase):
    def test_native_token_counts(self):
        self.assertEqual(token_counts(torch.tensor([600, 800, 900, 1000])).tolist(), [12, 15, 15, 18])

    def test_pooling_retains_channels_and_ignores_padding(self):
        tokens = torch.full((2, 9, 2), 1e6)
        tokens[0, :6] = torch.arange(6.)[:, None]
        tokens[1] = torch.arange(9.)[:, None]
        result = pool_tokens(tokens, torch.tensor([6, 9]))
        torch.testing.assert_close(result[0], torch.tensor([.5, .5, 2.5, 2.5, 4.5, 4.5]))
        torch.testing.assert_close(result[1], torch.tensor([1., 1., 4., 4., 7., 7.]))

    def test_answer_and_eeg_positions(self):
        hidden = torch.arange(14.).view(1, 14, 1).expand(2, -1, 2)
        cap = {"reve": hidden[:, :9], "connector": hidden[:, :9], "hidden": hidden}
        out = collect_stages(cap, torch.tensor([6, 9]), 2, 3)
        torch.testing.assert_close(out["qwen_answer"], torch.tensor([[10., 10.], [13., 13.]]))
        torch.testing.assert_close(out["qwen_eeg"][0], torch.tensor([2.5, 2.5, 4.5, 4.5, 6.5, 6.5]))

    def test_equal_dataset_class_mass(self):
        rows = [{"dataset": "A", "label": 0}]*3 + [{"dataset": "A", "label": 1}] + [
                {"dataset": "B", "label": 0}, {"dataset": "B", "label": 1}]
        weights = group_weights(rows)
        self.assertAlmostEqual(weights.mean(), 1.)
        np.testing.assert_allclose([weights[:3].sum(), *weights[3:]], 1.5)

    def test_convex_probe_converges(self):
        x = torch.tensor([[-2., 1.], [-1., -1.], [1., -1.], [2., 1.]], dtype=torch.float64)
        y = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
        w, b, record = fit_logistic(x, y, torch.ones_like(y), .1)
        self.assertLess(record["gradient_max_abs"], 1e-5)
        self.assertEqual(((x @ w + b) > 0).int().tolist(), [0, 0, 1, 1])

    def test_auc_and_prediction_metrics(self):
        rows = [{"dataset_index": 0, "label": i} for i in [0, 1, 0, 1]]
        result = metrics_from_scores([-2, 2, -1, 1], rows, ["A"])
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["roc_auc"], 1.)


if __name__ == "__main__":
    unittest.main()
