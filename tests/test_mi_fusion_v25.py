import numpy as np
import pytest
from eeg_tasks.mi_fusion_v25 import mix


def test_probability_endpoints_and_class_order():
    a = np.array([[.2, .8], [.9, .1]])
    b = np.array([[.6, .4], [.1, .9]])
    np.testing.assert_array_equal(mix(a, b, 0), a)
    np.testing.assert_array_equal(mix(a, b, 1), b)
    np.testing.assert_allclose(mix(a, b, .25).sum(1), 1)
    np.testing.assert_allclose(mix(a, b, .25), [[.3, .7], [.7, .3]])
    with pytest.raises(ValueError):
        mix(a, b, 1.1)
