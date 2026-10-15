import numpy as np
import pytest
from eeg_tasks.mi_covariance_v23 import covariance_vectors


def test_global_gain_invariance_and_electrode_gain_difference():
    x = np.random.default_rng(42).normal(size=(3, 600))
    base = covariance_vectors(x)
    scaled = covariance_vectors(x * 1e-6)
    electrode = covariance_vectors(x * np.array([2., 1., 1.])[:, None])
    for arm in base:
        np.testing.assert_allclose(base[arm], scaled[arm], atol=1e-12)
    np.testing.assert_allclose(base['correlation'], electrode['correlation'], atol=1e-12)
    assert np.linalg.norm(base['relative_covariance'] - electrode['relative_covariance']) > .5


def test_flat_and_nonfinite_rejected():
    x = np.ones((3, 600))
    with pytest.raises(ValueError):
        covariance_vectors(x)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        covariance_vectors(x)
