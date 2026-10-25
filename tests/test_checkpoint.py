import copy
import random
import numpy as np
import pytest
import torch
from eeg_align.checkpoint import capture_model,restore_model,capture_rng,restore_rng,seed_rng,recover_frozen_warmup
from eeg_align.normalization import apply_bn_policy
from eeg_align.distributed import atomic_save


def net():
    return torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.BatchNorm1d(6),torch.nn.Dropout(.25),torch.nn.Linear(6,2))


def test_checkpoint_restores_buffers_predictions_rng_and_next_update(tmp_path):
    seed_rng(7);model=net();opt=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.randn(12,4);y=torch.arange(12)%2
    for _ in range(3):
        opt.zero_grad();torch.nn.functional.cross_entropy(model(x),y).backward();opt.step()
    model.eval();expected=model(x).detach().clone();model.train()
    state=dict(**capture_model(model),optimizer=copy.deepcopy(opt.state_dict()),rng=capture_rng())
    path=tmp_path/'state.pt';atomic_save(state,path)
    opt.zero_grad();torch.nn.functional.cross_entropy(model(x),y).backward();opt.step()
    next_state=capture_model(model)
    expected_random=(random.random(),np.random.random(),torch.rand(3))
    restored=net();optimizer=torch.optim.AdamW(restored.parameters(),lr=.01)
    saved=torch.load(path,weights_only=False);restore_model(restored,saved)
    restored.eval();torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)
    optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);restored.train()
    optimizer.zero_grad();torch.nn.functional.cross_entropy(restored(x),y).backward();optimizer.step()
    actual=capture_model(restored)
    for kind in ['trainable','buffers']:
        for k,v in next_state[kind].items():torch.testing.assert_close(actual[kind][k],v,rtol=0,atol=0)
    assert random.random()==expected_random[0]
    assert np.random.random()==expected_random[1]
    torch.testing.assert_close(torch.rand(3),expected_random[2],rtol=0,atol=0)
    del saved['buffers']
    with pytest.raises(ValueError):restore_model(restored,saved)


def test_frozen_bn_statistics_keep_encoder_and_affine_gradients():
    seed_rng(11);model=net();model.train();apply_bn_policy(model,'frozen')
    before={n:b.clone() for n,b in model.named_buffers()}
    out=model(torch.randn(12,4)+10);out.square().mean().backward()
    assert model[2].training  # Other dropout is NOT disabled.
    assert model[0].weight.grad.abs().sum()>0
    assert model[1].weight.grad.abs().sum()>0
    for n,b in model.named_buffers():torch.testing.assert_close(b,before[n],rtol=0,atol=0)
    model.train();apply_bn_policy(model,'updating');model(torch.randn(12,4)+10)
    assert model[1].num_batches_tracked>before['1.num_batches_tracked']


def test_legacy_recovery_refuses_unfrozen_or_changed_encoder():
    model=torch.nn.Module();model.add_module('eeg_encoder',net());model.add_module('head',torch.nn.Linear(2,2))
    state=capture_model(model);state.pop('buffers');state.pop('format_version');state['step']=100
    assert recover_frozen_warmup(model,state,100)['unchanged_encoder_tensors']>0
    state['step']=101
    with pytest.raises(ValueError):recover_frozen_warmup(model,state,100)
    state['step']=100;state['trainable']['eeg_encoder.0.weight'].add_(1)
    with pytest.raises(ValueError):recover_frozen_warmup(model,state,100)
