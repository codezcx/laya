import pytest
import torch
from eeg_tasks.emotion_alignment import cross_source_loss

def test_alignment_rewards_same_class_across_sources():
 labels=torch.tensor([0,1,2,0,1,2]);domains=torch.tensor([0,0,0,1,1,1]);good=torch.eye(3).repeat(2,1).requires_grad_();bad=good.detach().clone();bad[3:]=bad[3:].roll(1,0)
 assert cross_source_loss(good,labels,domains)<cross_source_loss(bad,labels,domains)
 loss=cross_source_loss(good,labels,domains);loss.backward();assert torch.isfinite(good.grad).all()

def test_alignment_requires_cross_source_positive():
 with pytest.raises(ValueError):cross_source_loss(torch.randn(3,4),torch.tensor([0,1,2]),torch.zeros(3,dtype=torch.long))
