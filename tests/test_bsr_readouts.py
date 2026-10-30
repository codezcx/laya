import pytest
import torch
from eeg_align.token_resampler import TokenReadout, KINDS
from eeg_align.online_bsr import compatible_encoder_state
from eeg_align.lr_schedule import rates_at
from eeg_align.task_options import make_variant
from eeg_align.bsr_data import pad_for_stft
import numpy as np


def test_short_p300_stft_padding_does_not_crop_or_rescale_time():
    x=np.arange(161)[None,None,:]
    y=pad_for_stft(x)
    assert y.shape[-1]==165
    np.testing.assert_array_equal(x,y[...,:161])
    assert 1+y.shape[-1]//11==16
    for length in [201,601,800,831,1001,1500,2000]:
        x=np.ones((1,1,length))
        assert pad_for_stft(x).shape==x.shape


@pytest.mark.parametrize('kind', KINDS)
def test_readout_shapes_masks_and_gradients(kind):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = TokenReadout(kind, width=24, heads=4, queries=3, depth=2)
    x = torch.randn(2, 9, 24, requires_grad=True)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[:, -2:] = False
    y, m = model(x, valid)
    assert y.shape == (2, 3 if kind in {'cross_attention','qformer'} else 1, 24)
    assert m.all()
    dirty = x.detach().clone(); dirty[:, -2:] = float('nan')
    torch.testing.assert_close(y, model(dirty, valid)[0])
    y[..., 0].sum().backward()
    assert x.grad[:, :-2].abs().sum() > 0
    assert x.grad[:, -2:].abs().sum() == 0
    if kind != 'mean_pool':
        assert model.query.grad.abs().sum() > 0
    with pytest.raises(ValueError): model(x, torch.zeros_like(valid))


def test_old_bias_conversion_exact_projections():
    torch.manual_seed(8)
    x = torch.randn(3, 5)
    wq, wk, wv = [torch.randn(7,5) for _ in range(3)]
    q,v = torch.randn(7), torch.randn(7)
    old = {'a.q_bias': q, 'a.v_bias': v, 'a.query.weight': wq}
    new = compatible_encoder_state(old, {'a.query.bias': q})
    assert 'a.q_bias' in old and 'a.q_bias' not in new
    for weight, old_bias, key in [(wq,q,'query'),(wk,None,'key'),(wv,v,'value')]:
        torch.testing.assert_close(torch.nn.functional.linear(x,weight,old_bias),
            torch.nn.functional.linear(x,weight,new['a.'+key+'.bias']),rtol=0,atol=0)


def test_two_phase_schedule():
    peaks={'connector':1e-4,'readout':1e-4,'jev_lora':5e-6,'decision_head':1e-5}
    assert rates_at(340,2040,340,peaks)['jev_lora']==0
    assert rates_at(341,2040,340,peaks)['jev_lora']>0
    for name,lr in rates_at(2040,2040,340,peaks).items():
        assert lr==pytest.approx(peaks[name]*.02)
    assert rates_at(62,2040,340,peaks)['connector']==peaks['connector']


def test_task_text_only_in_suffix_and_label_independent():
    task={'task_id':'fake','instruction':'Decode EEG','question':'Which class?', 'classes':[
        {'class_id':'0','option':'Left'},{'class_id':'1','option':'Right'}]}
    a=make_variant(task,0,'sample',1,42,None)
    b=make_variant(task,1,'sample',1,42,None)
    assert a['suffix']==b['suffix'] and a['label']!=b['label']
    assert a['suffix'].count(task['instruction'])==1
    assert a['suffix'].endswith('[decision]:')
