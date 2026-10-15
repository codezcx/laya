import numpy as np
from eeg_tasks.emotion_transfer_probe import fractions,canonical,transform,CHANNELS

def test_log_fraction_ignores_channel_gain():
 x=np.arange(2*30*5,dtype=float).reshape(2,30,5)/100
 offsets=np.random.default_rng(0).normal(size=(2,30,1))
 np.testing.assert_allclose(fractions(x),fractions(x+offsets),atol=1e-12)
 np.testing.assert_allclose(np.exp(fractions(x)).sum(-1),1)

def test_spatial_center_and_legacy_channel_aliases():
 assert canonical('t3')==canonical('T7')
 x=np.random.default_rng(1).normal(size=(2,30,5));offset=np.random.default_rng(2).normal(size=(2,1,5))
 np.testing.assert_allclose(transform(x,'spatial_linear'),transform(x+offset,'spatial_linear'),atol=1e-12)
 assert transform(x,'three_linear').shape==(2,15)
 assert len(set(map(canonical,CHANNELS)))==30
