import numpy as np
from eeg_tasks.emotion_reference import reference,spectrum

def test_car_removes_shared_reference_before_amplitude_scaling():
 rng=np.random.default_rng(1);x=rng.normal(size=(30,600));common=rng.normal(size=(1,600))
 np.testing.assert_allclose(reference(x,'car30'),reference(x+common,'car30'),atol=1e-14)
 np.testing.assert_allclose(reference(x,'car30').mean(0),0,atol=1e-14)
 np.testing.assert_array_equal(reference(x,'as_recorded'),x)

def test_band_fraction_is_common_unit_invariant():
 x=np.random.default_rng(2).normal(size=(30,600))
 np.testing.assert_allclose(spectrum(reference(x,'car30')),spectrum(reference(x*1e-6,'car30')),atol=1e-12)
