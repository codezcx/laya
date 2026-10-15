"""Small report-integrity checks, without loading pretrained models."""
import json
import tempfile
import unittest
from pathlib import Path

from eeg_align.report_reve_only import audit_predictions, exact_counts
from eeg_align.reve_baseline import classification_metrics


class ReveReportTest(unittest.TestCase):
    def fixture(self, root):
        truth = [{"path": f"trial{i}", "dataset": "A", "subject": 1,
                  "label": i, "dataset_index": 0} for i in range(2)]
        predictions = [{**row, "subject": "1", "prediction": 0,
                        "probabilities": [.7, .3]} for row in truth]
        paths = [root / name for name in ("manifest.jsonl", "predictions.jsonl", "metrics.json")]
        for path, rows in zip(paths, (truth, predictions)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[2].write_text(json.dumps(classification_metrics([0, 1], [0, 0], [0, 0], ["A"])))
        return paths, predictions

    def test_prediction_audit_recomputes_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = self.fixture(Path(directory))
            metrics, subjects, _ = audit_predictions(*paths, ["A"])
            self.assertEqual(metrics["accuracy"], .5)
            self.assertEqual(subjects, [{"dataset": "A", "subject": "1", "n": 2,
                                         "correct": 1, "accuracy": .5}])

    def test_prediction_audit_rejects_wrong_identity_and_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, predictions = self.fixture(Path(directory))
            predictions[0]["path"] = "not_this_trial"
            paths[1].write_text("".join(json.dumps(row) + "\n" for row in predictions))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                audit_predictions(*paths, ["A"])
            paths, predictions = self.fixture(Path(directory))
            predictions[0]["probabilities"] = [.1, .9]
            paths[1].write_text("".join(json.dumps(row) + "\n" for row in predictions))
            with self.assertRaisesRegex(ValueError, "argmax"):
                audit_predictions(*paths, ["A"])

    def test_previous_float32_accuracy_uses_exact_counts(self):
        metrics = {"n": 3, "accuracy": .666666687, "confusion_matrix": [[1, 0], [1, 1]]}
        self.assertEqual(exact_counts(metrics)["accuracy"], 2/3)
        metrics["accuracy"] = .8
        with self.assertRaises(ValueError):
            exact_counts(metrics)


if __name__ == "__main__":
    unittest.main()
