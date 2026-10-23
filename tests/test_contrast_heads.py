import torch
from eeg_align.contrast_heads import SplitContrastHeads


def test_identical_initialization_independent_parameters_and_routing():
    torch.manual_seed(42)
    initial = torch.nn.Linear(4, 3)
    rng = torch.get_rng_state().clone()
    heads = SplitContrastHeads(initial)
    assert torch.equal(rng, torch.get_rng_state())
    assert heads.mi.weight.data_ptr() != heads.seed.weight.data_ptr()
    x = torch.randn(3, 4, requires_grad=True)
    assert torch.allclose(heads(x, [1, 0, 2]), initial(x), atol=1e-6)
    z = heads(x, [1, 0, 2])
    z[1].sum().backward()
    assert heads.mi.weight.grad.abs().sum() > 0
    assert heads.seed.weight.grad.abs().sum() == 0
    assert x.grad[1].abs().sum() > 0
    assert x.grad[[0, 2]].abs().sum() == 0
