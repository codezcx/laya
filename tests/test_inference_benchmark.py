import numpy as np
import pytest

from scripts.benchmark_inference import duration_input, preprocess, summary


def test_duration_prefix_and_repeat_not_time_resampling():
    trial = np.arange(800, dtype=np.float32)[None,None,:]
    assert np.array_equal(duration_input(trial, .5), trial[:,:,:100])
    assert np.array_equal(duration_input(trial, 8), np.concatenate([trial,trial],axis=-1))
    with pytest.raises(ValueError):
        duration_input(trial, .1)


def test_chunk_lengths_and_token_counts():
    trial = np.random.default_rng(42).normal(size=(1,22,800)).astype(np.float32)
    spatial = np.random.default_rng(2).normal(size=(65,22)).astype(np.float32)
    for seconds, chunks in [(0.5,[100]),(4,[800]),(16,[1600,1600]),(30,[2000,2000,2000])]:
        pieces = preprocess(duration_input(trial, seconds),spatial)
        assert [x.shape[-1] for x in pieces] == chunks
        assert sum(2 * x.shape[-1]//100 for x in pieces) == int(4*seconds)
        assert all(x.shape[:2] == (1,65) and x.dtype == np.float32 for x in pieces)


def test_statistics_keep_raw_observations():
    result = summary([1,2,3])
    assert result['median_ms'] == 2 and result['n'] == 3
    assert result['samples_ms'] == [1,2,3]


def test_leaf_readout_single_chunk_is_upstream_and_long_is_explicit_weighted_mean():
    import torch
    from scripts.benchmark_inference import leaf_cosine_readout
    prototypes = torch.eye(3)
    first = torch.tensor([[.8,.5,.2]])
    second = torch.tensor([[.1,.7,.3]])
    assert torch.equal(leaf_cosine_readout([first],prototypes,[800]),first @ prototypes.T)
    torch.testing.assert_close(leaf_cosine_readout([first,second],prototypes,[100,300]),
                               first*.25+second*.75)
    with pytest.raises(ValueError):
        leaf_cosine_readout([first],prototypes,[0])
    with pytest.raises(ValueError):
        leaf_cosine_readout([],prototypes,[])


def test_speed_bridge_is_dimension_matched_and_reuses_exact_decision_forward():
    import torch
    from types import SimpleNamespace
    from scripts.benchmark_inference import SpeedOnlyBridge
    from eeg_align.jev import JEVBridge
    class TinyText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = torch.nn.Embedding(100,8)
        def get_input_embeddings(self):
            return self.embedding
        def forward(self,inputs_embeds,**kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))
    tokenizer = SimpleNamespace(encode=lambda text,**kwargs:[1,2])
    model = SpeedOnlyBridge(TinyText(),torch.randn(16,8),tokenizer)
    assert SpeedOnlyBridge.forward is JEVBridge.forward
    assert model.connector[1].in_features == 256
    assert model.connector[1].out_features == model.connector[3].out_features == 8
    assert all(not p.requires_grad for p in model.parameters())
    output = model(torch.randn(1,16,256),torch.ones(1,16,dtype=torch.bool),num_choices=4)
    assert output['logits'].shape == (1,4) and torch.isfinite(output['logits']).all()
