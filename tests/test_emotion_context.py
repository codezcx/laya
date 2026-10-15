import numpy as np
import pytest
from eeg_tasks.emotion_context import bounds,spectrum

def test_context_stays_in_trial_and_contains_control():
 for total in [1800,2000,10000]:
  for start in [0,600,total-600]:
   lo,hi=bounds(start,1800,total);assert 0<=lo<=start and start+600<=hi<=total and hi-lo==1800
   assert bounds(start,600,total)==(start,start+600)
 with pytest.raises(ValueError):bounds(0,1800,1000)

def test_wider_band_keeps_high_frequency_signal():
 t=np.arange(1800)/200;x=np.tile(np.sin(2*np.pi*10*t)+np.sin(2*np.pi*35*t),(30,1))
 wide=spectrum(x,45);assert wide.shape==(30,6)
 assert (np.exp(wide[:,-1])>.4).all()
 assert spectrum(x[:,:600],30).shape==(30,5)
