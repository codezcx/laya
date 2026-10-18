import unittest
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scripts.probe_decision_state import fold_readout


class DecisionReadoutTest(unittest.TestCase):
    def test_exported_head_preserves_scaling_intercept_and_constant_feature(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(30,8))*np.arange(1,9)+np.arange(10,18)
        x[:,-1]=7
        y=np.eye(2)[np.arange(30)%2]
        model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,y)
        weight,bias=fold_readout(model)
        heldout=rng.normal(size=(10,8))+13;heldout[:,-1]=7
        np.testing.assert_allclose(heldout@weight.T+bias,model.predict(heldout),rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
