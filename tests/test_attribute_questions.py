import unittest
import numpy as np
from eeg_tasks.attribute_questions import relative_power, compare, measure, specifications


class AttributeQuestionsTest(unittest.TestCase):
    def test_frequency_and_scale(self):
        t = np.arange(600) / 200
        x = np.stack([np.sin(2*np.pi*10*t), np.sin(2*np.pi*20*t)])[None]
        p = relative_power(x)
        self.assertGreater(p['alpha'][0, 0], .99)
        self.assertGreater(p['beta'][0, 1], .99)
        for band, values in p.items():
            np.testing.assert_allclose(values, relative_power(x * 13)[band], atol=1e-12)

    def test_spatial_and_temporal_evidence(self):
        t = np.arange(600) / 200
        # C3 initially alpha, later beta; C4 always alpha.
        c3 = np.sin(2*np.pi*np.where(t < 1.5, 10, 20)*t)
        c4 = np.sin(2*np.pi*10*t)
        x = np.stack([c3, c4])[None]
        p = {name: relative_power(x[..., sl]) for name, sl in
             [('full', slice(None)), ('early', slice(0, 300)), ('late', slice(300, 600))]}
        specs = {s['id']: s for s in specifications()}
        self.assertEqual(measure(p, ['C3', 'C4'], specs['mi_spatial_alpha'])[-1][0], 'C')
        self.assertEqual(measure(p, ['C3', 'C4'], specs['mi_temporal_c3_alpha'])[-1][0], 'C')
        self.assertEqual(measure(p, ['C3', 'C4'], specs['mi_temporal_c4_alpha'])[-1][0], 'B')

    def test_tolerance_and_invalid_signal(self):
        _, answer = compare([2., 1.1, .5], [1., 1., 1.])
        self.assertEqual(answer.tolist(), ['A', 'B', 'C'])
        with self.assertRaises(ValueError):
            relative_power(np.zeros((1, 3, 600)))
        with self.assertRaises(ValueError):
            relative_power(np.full((1, 3, 600), np.nan))

    def test_catalog(self):
        specs = specifications()
        self.assertEqual(len({s['id'] for s in specs}), 12)
        self.assertEqual(sum(s['task'] == 'mi' for s in specs), 6)


if __name__ == '__main__':
    unittest.main()
