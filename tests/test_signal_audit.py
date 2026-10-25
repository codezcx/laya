import numpy as np
from scripts.audit_benchmark import frequency_scores


def test_frequency_control_recovers_all_candidates_despite_phase_gain_offset():
    f=np.arange(9.25,15,.5);t=np.arange(831)/200
    phase=np.arange(8)[None,:,None]*.7
    x=np.sin(2*np.pi*f[:,None,None]*t[None,None,:]+phase)
    expected=np.arange(len(f))
    np.testing.assert_array_equal(frequency_scores(x,200,f).argmax(-1),expected)
    altered=x*np.arange(1,9)[None,:,None]+np.arange(8)[None,:,None]*100
    np.testing.assert_array_equal(frequency_scores(altered,200,f).argmax(-1),expected)
