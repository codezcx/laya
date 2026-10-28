import io
from pathlib import Path

import pytest
import torch
from torch import nn

from eeg_align.token_resampler import (
    ResamplerConfig, ResampledDecisionModel, TokenResampler,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


@pytest.fixture(params=["cross_attention", "qformer"])
def model(request):
    torch.manual_seed(7)
    return TokenResampler(ResamplerConfig(
        kind=request.param, input_dim=24, hidden_dim=24, num_heads=4,
    )).eval()


def test_default_recipes():
    path = Path(__file__).resolve().parents[1] / "configs/token_resamplers.json"
    for kind, depth in [("cross_attention", 1), ("qformer", 4)]:
        config = ResamplerConfig.from_json(path, kind)
        assert config.num_queries == 32
        assert config.input_dim == config.hidden_dim == 768
        assert config.depth == depth
        assert ResamplerConfig(**config.to_dict()) == config


@pytest.mark.parametrize("length", [1, 19, 73])
def test_shapes_and_input_dependence(model, length):
    x = torch.randn(2, length, 24)
    y, valid = model(x)
    assert y.shape == (2, 32, 24)
    assert valid.shape == (2, 32) and valid.dtype == torch.bool and valid.all()
    assert torch.isfinite(y).all()
    assert not torch.allclose(y, model(torch.randn_like(x))[0])


def test_padding_and_batch_independence(model):
    x = torch.randn(2, 11, 24)
    valid = torch.arange(11)[None] < torch.tensor([5, 11])[:, None]
    x[0, 5:] = float("nan")
    y, _ = model(x, valid)
    torch.testing.assert_close(y[:1], model(x[:1, :5])[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y[1:], model(x[1:])[0], atol=1e-5, rtol=1e-5)
    padded = torch.cat([x, torch.full((2, 3, 24), float("inf"))], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(y, model(padded, padded_valid)[0], atol=1e-5, rtol=1e-5)


def test_gradients_reach_inputs_and_all_parameters(model):
    x = torch.randn(2, 13, 24, requires_grad=True)
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[0, 8:] = False
    y, _ = model(x, valid)
    (y - torch.randn_like(y)).square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name
    assert x.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(x.grad[~valid]) == 0


def test_rejects_invalid_inputs(model):
    with pytest.raises(ValueError, match="at least one"):
        model(torch.randn(2, 7, 24), torch.zeros(2, 7, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        model(torch.randn(2, 7, 24), torch.ones(2, 7))
    with pytest.raises(ValueError, match="bool"):
        model(torch.randn(2, 7, 24), torch.ones(2, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(2, 7, 25))
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.empty(2, 0, 24))
    with pytest.raises(ValueError, match="floating point"):
        model(torch.ones(2, 7, 24, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((2, 7, 24), float("nan")))


def test_checkpoint_round_trip(model):
    x = torch.randn(2, 11, 24)
    stream = io.BytesIO()
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored = TokenResampler(ResamplerConfig(**state["config"])).eval()
    restored.load_state_dict(state["state_dict"], strict=True)
    torch.testing.assert_close(restored(x)[0], model(x)[0], rtol=0, atol=0)


def test_bfloat16_encoder_output_and_autocast(model):
    x = torch.randn(2, 9, 24).bfloat16().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, _ = model(x)
        loss = (y.float() - torch.randn_like(y)).square().mean()
    loss.backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_projection_for_other_encoders():
    model = TokenResampler(ResamplerConfig(input_dim=16, hidden_dim=24, num_heads=4))
    x = torch.randn(2, 10, 16)
    y, _ = model(x)
    assert y.shape == (2, 32, 24)
    (y - torch.randn_like(y)).square().mean().backward()
    assert model.input_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    {"kind": "pool"}, {"num_queries": 0}, {"depth": 0}, {"depth": 2},
    {"num_heads": 5}, {"dropout": 1.0}, {"input_dim": True},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ResamplerConfig(**kwargs)


def test_decision_wrapper_registers_and_trains_resampler(model):
    class DecisionStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(24, 2)

        def forward(self, tokens, valid, labels=None):
            assert tokens.shape[1] == 32 and valid.all()
            logits = self.head(tokens.mean(dim=1))
            return {"logits": logits, "loss": nn.functional.cross_entropy(logits, labels)}

    wrapped = ResampledDecisionModel(model, DecisionStub())
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    before = model.queries.detach().clone()
    out = wrapped(torch.randn(2, 50, 24), labels=torch.tensor([0, 1]))
    assert out["logits"].shape == (2, 2)
    out["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, model.queries)
    assert "resampler.queries" in wrapped.state_dict()


def test_jev_assembly_keeps_all_32_queries(model):
    from eeg_align.jev import assemble_embeddings

    y, valid = model(torch.randn(2, 51, 24))
    prefix, suffix = torch.randn(3, 24), torch.randn(5, 24)
    assembled, mask, positions, last = assemble_embeddings(prefix, y, suffix, valid)
    assert assembled.shape == (2, 40, 24)
    assert mask.all() and last.tolist() == [39, 39]
    torch.testing.assert_close(assembled[:, 3:35], y)
    assert positions[0].tolist() == list(range(40))
