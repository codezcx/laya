import numpy as np
from eeg_tasks.data import normalized_window
from eeg_tasks.mi_reve_scaling_v24 import scale_window


def test_old_pipeline_exact_and_common_gain_invariance():
    x = np.random.default_rng(42).normal(size=(3, 600)) * np.array([1., 2., 3.])[:, None]
    np.testing.assert_array_equal(scale_window(x, 'per_channel'), normalized_window(x))
    common = scale_window(x, 'common_scale')
    np.testing.assert_allclose(common.std(-1) / common.std(-1)[0], x.std(-1) / x.std(-1)[0], rtol=1e-6)
    np.testing.assert_allclose(common, scale_window(x * 1e-6, 'common_scale'), atol=1e-6)
    np.testing.assert_allclose(scale_window(x, 'per_channel').std(-1), 1., atol=1e-6)
