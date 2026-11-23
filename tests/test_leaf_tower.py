"""Exercise the actual local release without loading JEV or accessing test EEG."""
import copy
import sys
import numpy as np
import torch
from eeg_align.online_leaf import OnlineLEAFTower, encode_batch, upstream_files
from eeg_align.runtime import CONFIG
from eeg_align.checkpoint import tensor_fingerprint


def test_release_tower_shapes_and_independence_from_text():
    torch.set_num_threads(2)
    model = OnlineLEAFTower().eval()
    assert all(p.exists() for p in upstream_files())
    assert not any('qformer' in n or 'film' in n or 'outProj' in n for n,_ in model.named_parameters())
    from LEAF import Tower
    from load_config import build_model_config, load_yaml
    reference = Tower(build_model_config(load_yaml(CONFIG['leaf_root']+'/configs/LEAF_mpnet.yaml'))).eval()
    state = torch.load(CONFIG['checkpoint'], weights_only=True, map_location='cpu')
    reference.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith('tower.')}, strict=True)
    with torch.no_grad():
        for length, count in ((800,16), (2000,40)):
            x = torch.randn(2,65,length)
            actual = model(x)
            assert actual.shape == (2,count,256)
            torch.testing.assert_close(actual, reference(x), rtol=0, atol=0)
    before = tensor_fingerprint(dict(model.named_buffers()))
    model.train()
    model(torch.randn(2,65,200)).square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert before == tensor_fingerprint(dict(model.named_buffers()))


def test_all_chunks_and_mask_preserved():
    class Encoder(torch.nn.Module):
        def forward(self,x):
            return x[:,0,::100,None].repeat(1,2,3)
    a = np.ones((1,65,200), dtype=np.float32)
    b = np.full((1,65,100),2,dtype=np.float32)
    tokens, mask = encode_batch(Encoder(), [(0,[1],[a,b]),(0,[0],[b])], 2)
    assert tokens.shape == (2,6,3)
    assert mask.sum(1).tolist() == [2,6]
    assert tokens[1,:,0].tolist() == [1,1,1,1,2,2]
    assert tokens[0,:2,0].tolist() == [2,2]
