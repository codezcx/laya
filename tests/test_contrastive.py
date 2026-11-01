import math
import pytest
import torch
import torch.nn.functional as F

from eeg_contrastive.objectives import positive_mask, per_sample_loss, class_scores
from eeg_contrastive.protocol import build_text_manifest, rates_at, CONFIG
from eeg_align.benchmark_data import catalog


def test_semantics_across_label_orders_and_datasets():
    bank = build_text_manifest()
    hmc, edf = bank['banks']['hmc'], bank['banks']['SleepEDF']
    assert hmc['semantic_ids'][1] == edf['semantic_ids'][4]  # REM
    assert hmc['semantic_ids'][2] == edf['semantic_ids'][1]  # N1
    assert hmc['semantic_ids'][4] == edf['semantic_ids'][3]  # N3
    assert bank['banks']['bcic_2a']['semantic_ids'][1] == bank['banks']['AlexMI']['semantic_ids'][0]
    assert bank['banks']['workload']['semantic_ids'][0] != bank['banks']['AlexMI']['semantic_ids'][2]
    assert bank['banks']['seed']['semantic_ids'][0] != bank['banks']['seed_vii']['semantic_ids'][2]


def test_candidate_banks_have_only_task_classes_and_all_templates():
    bank = build_text_manifest()
    assert len(bank['texts']) == len(set(bank['texts']))
    for task in catalog():
        item = bank['banks'][task['task_id']]
        labels = torch.tensor(item['class_labels'])
        assert len(labels) == len(task['classes']) * 3
        mask = positive_mask(torch.arange(len(task['classes'])), labels)
        assert mask.sum(1).eq(3).all()
        for c in range(len(task['classes'])):
            assert all(bank['texts'][item['text_indices'][j]].startswith(task['instruction']) for j in torch.where(labels==c)[0])


def test_duplicate_same_class_eegs_are_not_negatives():
    mask = positive_mask(torch.tensor([0,0,1]), torch.tensor([0,0,0,1,1,1]))
    assert torch.equal(mask[0], mask[1])
    assert mask[0,:3].all() and not mask[0,3:].any()


def test_one_text_infonce_is_exactly_classification_cross_entropy():
    x = torch.randn(6,4,requires_grad=True)
    y = torch.tensor([0,1,2,3,0,1])
    actual = per_sample_loss(x, positive_mask(y,torch.arange(4)), 'infonce')
    torch.testing.assert_close(actual,F.cross_entropy(x,y,reduction='none'))


def test_duplicated_templates_add_only_log_count_to_infonce():
    x = torch.tensor([[1.,2.,-1.],[3.,0.,2.]],requires_grad=True)
    labels = torch.tensor([1,0])
    repeated = x.repeat_interleave(3,1)
    actual=per_sample_loss(repeated,positive_mask(labels,torch.arange(3).repeat_interleave(3)),'infonce')
    expected=F.cross_entropy(x,labels,reduction='none')+math.log(3)
    torch.testing.assert_close(actual,expected)
    g1=torch.autograd.grad(actual.mean(),x,retain_graph=True)[0]
    g2=torch.autograd.grad(expected.mean(),x)[0]
    torch.testing.assert_close(g1,g2)


@pytest.mark.parametrize('kind',['infonce','sigmoid'])
def test_loss_decreases_when_all_positives_rise_and_negatives_fall(kind):
    mask=positive_mask(torch.tensor([0,1]),torch.tensor([0,0,1,1]))
    logits=torch.zeros(2,4,requires_grad=True)
    loss=per_sample_loss(logits,mask,kind).mean()
    grad=torch.autograd.grad(loss,logits)[0]
    assert (grad[mask]<0).all() and (grad[~mask]>0).all()
    improved=torch.where(mask,torch.tensor(3.),torch.tensor(-3.))
    assert per_sample_loss(improved,mask,kind).mean()<loss


def test_sigmoid_balancing_is_invariant_to_number_of_equivalent_negatives():
    x=torch.tensor([[2.,-1.]])
    many=torch.tensor([[2.,-1.,-1.,-1.,-1.]])
    a=per_sample_loss(x,torch.tensor([[True,False]]),'sigmoid')
    b=per_sample_loss(many,torch.tensor([[True,False,False,False,False]]),'sigmoid')
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('kind',['infonce','sigmoid'])
def test_microbatch_accumulation_matches_full_batch(kind):
    torch.manual_seed(42)
    x=torch.randn(8,4,requires_grad=True)
    y=torch.arange(8)%2
    mask=positive_mask(y,torch.tensor([0,0,1,1]))
    g=torch.autograd.grad(per_sample_loss(x,mask,kind).mean(),x)[0]
    for start in range(0,8,2):
        (per_sample_loss(x[start:start+2],mask[start:start+2],kind).sum()/8).backward()
    torch.testing.assert_close(g,x.grad)


def test_scores_use_same_template_average_and_loss_rejects_invalid_masks():
    x=torch.tensor([[1.,3.,2.,0.]])
    torch.testing.assert_close(class_scores(x,torch.tensor([0,0,1,1]),2),torch.tensor([[2.,1.]]))
    for y in [torch.tensor([5]),torch.tensor([0])]:
        with pytest.raises(ValueError):positive_mask(y,torch.tensor([0,0]))
    with pytest.raises(ValueError):per_sample_loss(x,torch.ones_like(x,dtype=torch.bool),'infonce')
    assert 0<rates_at(1,2040)['encoder']<CONFIG['learning_rates']['encoder']
    assert rates_at(2040,2040)['encoder']==pytest.approx(CONFIG['learning_rates']['encoder']*.02)
