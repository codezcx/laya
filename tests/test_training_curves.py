import numpy as np

from scripts.plot_training_curves import padded_limits, trailing_mean, validation_series


def test_trailing_mean_requires_full_window_and_preserves_gaps():
    result = trailing_mean([1, 2, 3, np.nan, 5, 6, 7], 3)
    np.testing.assert_allclose(result, [np.nan, np.nan, 2, np.nan, np.nan, np.nan, 6], equal_nan=True)


def test_axis_range_keeps_all_observations_visible():
    lo, hi = padded_limits([30, 40, 50], lower=0, upper=100)
    assert 0 <= lo < 30 and 50 < hi <= 100
    lo, hi = padded_limits([0, 100], lower=0, upper=100)
    assert (lo, hi) == (0, 100)


def test_validation_keeps_step_zero_and_missing_evaluations_as_gaps():
    report = dict(step=0, macro_balanced_accuracy=.5,
                  datasets={'a': dict(accuracy=.2, balanced_accuracy=.3),
                            'b': dict(accuracy=.6, balanced_accuracy=.7)})
    acc, balanced = validation_series([report], [0, 340])
    np.testing.assert_allclose(acc, [40, np.nan], equal_nan=True)
    np.testing.assert_allclose(balanced, [50, np.nan], equal_nan=True)
