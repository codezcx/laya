import numpy as np
import torch
from eeg_align.tuab_data import canonical,window_starts,split_records,selected_windows,metrics
from eeg_align.train_jev import record_loss,backward_bags


def test_channels_and_nonoverlapping_windows():
    assert canonical('EEG T3-REF')=='T7'
    assert canonical('EEG FP1-REF')=='FP1'
    for duration,count in [(100,8),(200,16),(1800,16)]:
        starts=window_starts(duration,count)
        assert min(starts)>=0 and max(starts)+10<=min(duration,1200)
        assert np.min(np.diff(starts))>=10-1e-8


def test_subject_split_keeps_mixed_labels_together():
    rows=[dict(subject=f'p{i}',label=j,official_split='train') for i in range(30) for j in [0,1]]
    rows += [dict(subject='heldout',label=0,official_split='eval')]
    out=split_records(rows)
    for subject in {r['subject'] for r in out}:
        assert len({r['split'] for r in out if r['subject']==subject})==1
    assert out[-1]['split']=='test'


def test_temporal_sampling_is_reproducible_and_label_independent():
    r=dict(id='example',split='train',label=0)
    a=selected_windows(r,2);b=selected_windows(dict(r,label=1),2)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a//2,np.arange(8))
    assert any(not np.array_equal(a,selected_windows(r,e)) for e in range(10))
    np.testing.assert_array_equal(selected_windows(dict(r,split='validation'),999),np.arange(8))


class Toy(torch.nn.Module):
    def __init__(self):super().__init__();self.layer=torch.nn.Linear(3,2).double()
    def forward(self,x,valid):return dict(logits=self.layer(x.mean(1)))


def test_two_pass_gradient_matches_whole_bag_autograd():
    torch.manual_seed(8);model=Toy();x=torch.randn(24,5,3,dtype=torch.float64);labels=torch.tensor([0,1,0])
    loss=record_loss(model(x,None)['logits'],labels);(loss*.7).backward()
    expected=[p.grad.clone() for p in model.parameters()];model.zero_grad()
    actual,_=backward_bags(model,x,labels,microbatch=5,scale=.7)
    assert abs(actual-loss.item())<1e-10
    for p,grad in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,grad)


def test_probability_tie_matches_first_logit_and_record_metrics():
    m=metrics([0,1,1],[.5,.9,.2])
    assert m['confusion']==[[1,0],[1,1]]
    assert m['balanced_accuracy']==.75
