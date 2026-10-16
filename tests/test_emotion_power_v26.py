import numpy as np
from eeg_tasks.emotion_power_v26 import power_features
from eeg_tasks.emotion_reference import spectrum


def test_exact_control_and_spatial_amplitude_information():
    x = np.random.default_rng(42).normal(size=(30, 600))
    np.testing.assert_array_equal(power_features(x, 'per_channel'), spectrum(x))
    gain = np.linspace(1, 3, 30)[:, None]
    np.testing.assert_allclose(power_features(x, 'per_channel'), power_features(x * gain, 'per_channel'), atol=1e-12)
    assert np.linalg.norm(power_features(x, 'joint_power') - power_features(x * gain, 'joint_power')) > 1
    for mode in ['per_channel', 'joint_power']:
        np.testing.assert_allclose(power_features(x, mode), power_features(x * 1e-6, mode), atol=1e-12)
