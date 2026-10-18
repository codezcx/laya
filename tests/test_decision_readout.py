import unittest
import numpy as np
import tempfile
from pathlib import Path
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scripts.probe_decision_state import fold_readout
from eeg_align.readout import describe_head,FrozenReadout,pool_reve


class DecisionReadoutTest(unittest.TestCase):
    def test_pool_retains_segment_and_channel_order(self):
        x=torch.arange(18).reshape(1,18,1).expand(1,18,512)
        np.testing.assert_array_equal(pool_reve(x).numpy()[0,::512],[1,4,7,10,13,16])
        with self.assertRaises(ValueError):pool_reve(x[:,:12])

    def test_bound_head_checks_checkpoint_and_combined_feature_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'head.npz';w=np.zeros((2,8192));w[0,0]=2;w[1,5120]=3
            np.savez(path,weight=w,bias=np.array([1.,-1.]))
            protocol=dict(checkpoint_sha256='checkpoint',source_cache_sha256='cache',source_data_sha256='data')
            meta=describe_head(path,'combined',protocol,1000,3200)
            head=FrozenReadout(meta,'checkpoint','cache','data')
            states=np.ones((1,5120));reve=np.ones((1,3072))*2
            np.testing.assert_array_equal(head.score(states,reve),[[3,5]])
            with self.assertRaisesRegex(ValueError,'checkpoint_sha256'):FrozenReadout(meta,'other','cache','data')
            with self.assertRaises(ValueError):head.score(None,reve)

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
