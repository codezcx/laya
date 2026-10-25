import copy
import pytest
import torch
from eeg_align.lr_schedule import GroupLRSchedule


def setup_schedule(kind='warmup_cosine'):
    names=['connector','jev_lora','decision_head','eeg_encoder']
    peaks=[1e-4,5e-6,1e-5,1e-6]
    params=[torch.nn.Parameter(torch.tensor([1.])) for _ in names]
    opt=torch.optim.AdamW([dict(name=n,params=[p],lr=lr) for n,p,lr in zip(names,params,peaks)],weight_decay=0.)
    scheduler=GroupLRSchedule(opt,3000,90,.02,100,kind)
    return params,opt,scheduler


def test_first_peak_and_last_update_rates():
    _,_,s=setup_schedule()
    assert s.set_step(1)==pytest.approx(dict(connector=1e-4/90,jev_lora=5e-6/90,decision_head=1e-5/90,eeg_encoder=0.))
    assert s.set_step(90)['connector']==pytest.approx(1e-4)
    assert s.set_step(100)['eeg_encoder']==0.
    assert s.set_step(101)['eeg_encoder']==pytest.approx(1e-6/90)
    assert s.set_step(190)['eeg_encoder']==pytest.approx(1e-6)
    assert s.set_step(3000)==pytest.approx(dict(connector=2e-6,jev_lora=1e-7,decision_head=2e-7,eeg_encoder=2e-8))
    trace=[s.set_step(i)['connector'] for i in range(90,3001)]
    assert all(a>=b for a,b in zip(trace,trace[1:]))
    with pytest.raises(ValueError):s.set_step(0)
    with pytest.raises(ValueError):s.set_step(3001)


def test_schedule_checkpoint_reproduces_next_optimizer_update():
    p,opt,s=setup_schedule()
    for step in range(1,103):
        s.set_step(step)
        for value in p:value.grad=torch.ones_like(value)
        opt.step();opt.zero_grad(set_to_none=True)
    saved_opt=copy.deepcopy(opt.state_dict());saved_schedule=copy.deepcopy(s.state_dict())
    q,restored,t=setup_schedule()
    with torch.no_grad():
        for a,b in zip(q,p):a.copy_(b)
    restored.load_state_dict(saved_opt);t.load_state_dict(saved_schedule)
    assert t.set_step(103)==s.set_step(103)
    for parameters,optimizer in [(p,opt),(q,restored)]:
        for value in parameters:value.grad=torch.ones_like(value)
        optimizer.step()
    for a,b in zip(p,q):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_constant_control_and_smoke_encoder_unfreezing():
    _,_,s=setup_schedule('constant')
    for step in [1,100,101,3000]:
        lr=s.set_step(step)
        assert lr['connector']==1e-4
        assert lr['eeg_encoder']==(0. if step<=100 else 1e-6)
    _,opt,_=setup_schedule()
    smoke=GroupLRSchedule(opt,2,1,.02,1)
    assert smoke.set_step(1)['eeg_encoder']==0.
    assert smoke.set_step(2)['eeg_encoder']==1e-6
