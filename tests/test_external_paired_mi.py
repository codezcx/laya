import unittest
from scripts.build_external_paired_mi import cue_and_bounds


class ExternalPairedTimingTest(unittest.TestCase):
    def test_known_trial_matches_existing_annotation(self):
        cue, bounds = cue_and_bounds(54350)
        self.assertAlmostEqual(cue / 250, 220.396)
        self.assertEqual(bounds, [(43579, 43979), (44179, 44579)])
        for (a, b), (start, stop) in zip(bounds, [(-2.5, -.5), (.5, 2.5)]):
            self.assertEqual(b-a, 400)
            self.assertLessEqual(abs(a/200 - 220.396 - start), .0025)
            self.assertLessEqual(abs(b/200 - 220.396 - stop), .0025)

    def test_matlab_start_and_native_rate_are_explicit(self):
        self.assertEqual(cue_and_bounds(1)[0], 750)
        with self.assertRaises(ValueError):
            cue_and_bounds(1, fs=200)


if __name__ == '__main__':
    unittest.main()
