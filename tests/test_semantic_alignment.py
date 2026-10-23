"""Semantic-loss invariants, including repeated-class positives."""
import math
import torch
from eeg_align.semantic_alignment import multipositive_loss


def test_all_same_class_does_not_repel_identical_signals():
    z=torch.tensor([[1.,0.]]*4,requires_grad=True)
    loss=multipositive_loss(z,torch.zeros(4,dtype=torch.long),torch.eye(2),distributed=False)
    assert abs(loss.item()-math.log(4))<1e-6
    loss.backward()
    assert torch.allclose(z.grad,torch.zeros_like(z.grad),atol=1e-6)


def test_correct_semantics_beats_inverted_and_is_order_invariant():
    text=torch.eye(2);y=torch.tensor([0,0,1,1]);z=text[y].clone().requires_grad_()
    loss=multipositive_loss(z,y,text,distributed=False)
    assert loss<multipositive_loss(z,1-y,text,distributed=False)
    order=torch.tensor([2,0,3,1])
    assert torch.allclose(loss,multipositive_loss(z[order],y[order],text,distributed=False))
    loss.backward();assert torch.isfinite(z.grad).all()


def _distributed_gradient_worker(rank, rendezvous):
    import torch.distributed as dist
    from eeg_align.distributed import average_gradients
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    torch.set_num_threads(1);torch.manual_seed(7)
    head=torch.nn.Linear(3,2,bias=False)
    reference=torch.nn.Linear(3,2,bias=False);reference.load_state_dict(head.state_dict())
    x=torch.tensor([[1.,2.,0.],[0.,1.,2.],[2.,0.,1.],[1.,0.,2.],[2.,1.,0.]])
    y=torch.tensor([0,0,1,1,0]);text=torch.eye(2)
    indices=torch.arange(len(x))[rank::2]
    loss=multipositive_loss(head(x[indices]),y[indices],text)
    loss.backward();average_gradients(head.parameters(),2)
    expected=multipositive_loss(reference(x),y,text,distributed=False);expected.backward()
    assert torch.allclose(loss,expected,atol=1e-5)
    assert torch.allclose(head.weight.grad,reference.weight.grad,atol=1e-5,rtol=1e-5)
    dist.destroy_process_group()


def test_distributed_gradient_matches_global_batch_with_unequal_shards(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_gradient_worker,args=(str(tmp_path/'gloo'),),nprocs=2,join=True)

def test_mi_only_control_preserves_half_weight_and_zero_emotion_gradients():
    from eeg_align.semantic_alignment import task_contrastive_loss
    torch.manual_seed(4)
    z=torch.randn(12,8,requires_grad=True);text=torch.randn(7,8)
    tasks=torch.tensor([0]*4+[1]*4+[2]*4);labels=torch.tensor([0,1,0,1,2,3,2,3,4,5,6,4])
    _,parts=task_contrastive_loss(z,labels,tasks,text,{0:.5,1:.25,2:.25},distributed=False)
    loss=.5*parts[0]
    reference=.5*multipositive_loss(z[:4],labels[:4],text,distributed=False)
    torch.testing.assert_close(loss,reference)
    loss.backward()
    assert z.grad[:4].abs().sum()>0
    assert z.grad[4:].abs().sum()==0

def test_mi_seed_control_preserves_weights_and_excludes_seven_class_gradients():
    from eeg_align.semantic_alignment import task_contrastive_loss
    torch.manual_seed(4)
    z=torch.randn(12,8,requires_grad=True);text=torch.randn(7,8)
    tasks=torch.tensor([0]*4+[1]*4+[2]*4);labels=torch.tensor([0,1,0,1,2,3,2,3,4,5,6,4])
    _,parts=task_contrastive_loss(z,labels,tasks,text,{0:.5,1:.25,2:.25},distributed=False)
    loss=.5*parts[0]+.25*parts[1]
    reference=.5*multipositive_loss(z[:4],labels[:4],text,distributed=False)+.25*multipositive_loss(z[4:8],labels[4:8],text,distributed=False)
    torch.testing.assert_close(loss,reference)
    loss.backward()
    assert z.grad[:4].abs().sum()>0 and z.grad[4:8].abs().sum()>0
    assert z.grad[8:].abs().sum()==0
