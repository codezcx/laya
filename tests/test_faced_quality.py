import numpy as np
import pytest
from eeg_tasks.faced_prepare import prepare_window
from eeg_tasks.data import normalized_window

def test_strict_still_rejects_and_optin_quarantines_flat_channel():
 x=np.random.default_rng(1).normal(size=(3,600))*1e-5;x[1]=0
 with pytest.raises(ValueError):prepare_window(x,['C3','PO3','C4'])
 signal,record=prepare_window(x,['C3','PO3','C4'],True)
 assert signal is None and record['channels']==['PO3'] and record['std_volts']==[0.]

def test_valid_window_identical_and_nonfinite_never_silently_dropped():
 x=np.random.default_rng(2).normal(size=(3,600))*1e-5
 signal,record=prepare_window(x,['C3','Cz','C4'],True)
 assert record is None;np.testing.assert_array_equal(signal,normalized_window(x))
 x[0,0]=np.nan
 with pytest.raises(ValueError):prepare_window(x,['C3','Cz','C4'],True)
