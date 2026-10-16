import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from eeg_probes.data import build_protocols, check_protocol, fit_fbcsp
from eeg_probes.train import fit_linear, fit_mlp, labels_for, sample_weights, scores


def synthetic_rows(dataset):
    rows = []
    for subject in range(1, 9):
        sessions = range(2) if dataset == 'Lee2019_MI' else range(1)
        runs = range(6) if dataset == 'Dreyer2023' else range(1)
        for session in sessions:
            for run in runs:
                for trial in range(100):
                    label = (trial // 50) if dataset == 'Cho2017' else trial % 2
                    uid = f'{dataset}/{subject}/{session}/{run}/{trial}'
                    rows.append(dict(dataset=dataset, subject=str(subject), session=str(session), run=str(run),
                                     trial_number=trial, event_number=trial, label=label,
                                     path=uid, signal_sha256=uid, split='validation' if subject <= 2 else 'train'))
    return rows


class ProbePartitionTest(unittest.TestCase):
    def test_subject_and_task_have_different_correct_partitions(self):
        for dataset in ['Cho2017', 'Lee2019_MI', 'Dreyer2023']:
            rows = synthetic_rows(dataset)
            protocols = build_protocols(rows)
            self.assertEqual(len(protocols), 4)
            fit = protocols[0]['train']
            for protocol in protocols:
                check_protocol(rows, protocol, fit)
                y, names = labels_for(rows, protocol['target'])
                for part in ['train', 'validation', 'test']:
                    self.assertEqual(set(y[protocol[part]]), set(range(len(names))))
            self.assertEqual({rows[i]['label'] for i in protocols[2]['train']}, {0})
            self.assertEqual({rows[i]['label'] for i in protocols[2]['test']}, {1})
            self.assertEqual({rows[i]['label'] for i in protocols[3]['train']}, {1})
            self.assertEqual({rows[i]['label'] for i in protocols[3]['test']}, {0})
            if dataset == 'Cho2017':
                self.assertEqual(protocols[1]['domain'], 'within_record_index_blocks_only')

    def test_rejects_trial_or_csp_fit_leakage(self):
        rows = synthetic_rows('Lee2019_MI')
        protocols = build_protocols(rows)
        bad = dict(protocols[1], test=protocols[1]['train'])
        with self.assertRaisesRegex(ValueError, 'Trial leakage'):
            check_protocol(rows, bad, protocols[0]['train'])
        with self.assertRaisesRegex(ValueError, 'CSP saw'):
            check_protocol(rows, protocols[1], protocols[1]['train'])

    def test_weights_balance_people_and_tasks(self):
        rows = [dict(subject='1', label=0)] * 6 + [dict(subject='1', label=1)] * 2
        weights = sample_weights(rows, np.arange(8))
        self.assertAlmostEqual(float(weights[:6].sum()), float(weights[6:].sum()), places=6)

    def test_identity_score_is_macro_recall(self):
        rows = [dict(subject='1')] * 9 + [dict(subject='2')]
        y = np.array([0] * 9 + [1])
        result = scores(rows, range(10), y, np.zeros(10, dtype=int), 'subject', True)
        self.assertEqual(result['accuracy'], .9)
        self.assertEqual(result['selection_score'], .5)
        self.assertEqual(result['subject_macro'], .5)


class ProbeEstimatorTest(unittest.TestCase):
    def test_heldout_signals_and_labels_do_not_change_csp_fit(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(24, 3, 600))
        y = np.arange(24) % 2
        first, fitted = fit_fbcsp(x, y, list(range(16)))
        modified = x.copy()
        modified[16:] *= 100
        changed_labels = y.copy()
        changed_labels[16:] = 1 - changed_labels[16:]
        second, again = fit_fbcsp(modified, changed_labels, list(range(16)))
        self.assertEqual(first['fbcsp14'].shape, (24, 14))
        self.assertEqual(first['fbcsp6'].shape, (24, 6))
        np.testing.assert_array_equal(first['fbcsp14'][:16], second['fbcsp14'][:16])
        np.testing.assert_array_equal(fitted['selected_indices'], again['selected_indices'])
        for a, b in zip(fitted['csps'], again['csps']):
            np.testing.assert_allclose(a.filters_, b.filters_)

    def test_linear_scaler_only_sees_train_and_mlp_runs(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(80, 6)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        w = np.ones(60, dtype=np.float32)
        score = lambda pred: float(np.mean(pred == y[60:]))
        with tempfile.TemporaryDirectory() as temp:
            linear = Path(temp) / 'linear'
            linear.mkdir()
            prob, pred, config = fit_linear(x[:60], y[:60], x[60:], y[60:], w, score, linear)
            self.assertEqual(prob(x[60:]).shape, (20, 2))
            import joblib
            model = joblib.load(linear / 'model.joblib')
            np.testing.assert_allclose(model[0].mean_, x[:60].mean(0), rtol=1e-5, atol=1e-6)
            mlp = Path(temp) / 'mlp'
            mlp.mkdir()
            prob, _, config = fit_mlp(x[:60], y[:60], x[60:], y[60:], w, score, 42, mlp, max_epochs=2)
            np.testing.assert_allclose(prob(x[60:]).sum(1), 1., atol=1e-6)
            self.assertTrue((mlp / 'model.pt').is_file())


if __name__ == '__main__':
    unittest.main()
