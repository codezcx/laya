import numpy as np
import pytest
from eeg_tasks.emotion_late_fusion import blend

def test_fusion_endpoints_preserve_models():
 p=np.array([[.2,.3,.5]]);q=np.array([[.8,.1,.1]])
 np.testing.assert_array_equal(blend(p,q,0),p);np.testing.assert_array_equal(blend(p,q,1),q)
 np.testing.assert_allclose(blend(p,q,.25),[[.35,.25,.4]])

def test_reject_logits_and_invalid_weight():
 with pytest.raises(ValueError):blend(np.array([[2.,1.,0.]]),np.array([[1.,0.,0.]]),.5)
 with pytest.raises(ValueError):blend(np.array([[1.,0.,0.]]),np.array([[1.,0.,0.]]),1.1)
