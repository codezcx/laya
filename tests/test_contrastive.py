import math
import pytest
import torch
import torch.nn.functional as F

from eeg_align.contrastive_objectives import positive_mask, per_sample_loss, class_scores
from eeg_align.contrastive_protocol import build_text_manifest, rates_at, CONFIG
from eeg_align.benchmark_data import catalog
from eeg_align.contrastive_protocol import load_text_bank, load_aligned_tower, DATA


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


def test_pinned_text_bank_is_current_and_has_no_eeg_annotations():
    bank, vectors = load_text_bank()
    assert vectors.shape == (147, 768)
    assert len(bank['concepts']) == 49
    assert 'trial' not in bank and 'subject' not in bank


def test_sigmoid_zero_bias_is_float_even_when_json_contains_integer(monkeypatch):
    from eeg_align import train_contrastive as training
    monkeypatch.setattr(training, 'OnlineLEAFTower', lambda _: torch.nn.Linear(256,256))
    monkeypatch.setitem(training.CONFIG, 'initial_sigmoid_bias', 0)
    model = training.AlignmentModel('sigmoid', torch.ones(3,768))
    assert model.bias.requires_grad and model.bias.dtype == torch.float32
    assert model.bias.item() == 0


def test_downstream_recipe_is_unchanged():
    from scripts.run_contrastive import verify_baseline_recipe
    proof = verify_baseline_recipe()
    assert proof['parameter_counts']['eeg_encoder'] == 19_557_632


def test_smaller_global_batch_is_exact_prefix():
    # Exercise the production sampler on a lightweight, train-only fixture.
    from eeg_align.multitask_data import TrialStore
    store = TrialStore.__new__(TrialStore)
    store.tasks = [{'task_id': str(i)} for i in range(17)]
    store.rows, store.pools = {}, {}
    for tid in range(17):
        store.rows[tid] = [{'id':f'{tid}:{i}', 'split':'train'} for i in range(12)]
        store.pools[tid] = {c:{'a':[c*4,c*4+1], 'b':[c*4+2,c*4+3]} for c in range(3)}
    for step in range(1, 2041):
        tid,a = store.sample(step, 16, 42)
        tid2,b = store.sample(step, 32, 42)
        assert tid == tid2 and a == b[:16]


def test_transfer_excludes_projection_and_rejects_split_or_weight_changes(tmp_path):
    import copy
    import json
    from eeg_align.checkpoint import capture_model, tensor_fingerprint
    from eeg_align.distributed import atomic_save
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.BatchNorm1d(2))
            self.projection = torch.nn.Linear(2,3)
    model = Fixture()
    cache = {'a':'index_hash'}
    identity = dict(arm='infonce', smoke=False, config=CONFIG, data_config=DATA,
                    cache_indexes=cache, bank_sha256=CONFIG['text_bank_sha256'])
    done = dict(smoke=False, test_evaluations=0, jev_loaded=False, encoder_updated=True,
                checkpoint_reload_verified=True, frozen_text_verified=True, updates=2040,
                selected_step=340, selected_encoder_sha256=tensor_fingerprint(model.encoder.state_dict()))
    (tmp_path/'config.json').write_text(json.dumps(identity))
    (tmp_path/'complete.json').write_text(json.dumps(done))
    state = dict(**capture_model(model), identity=identity, step=340)
    atomic_save(state, tmp_path/'best.pt')
    fresh = Fixture()
    projection = tensor_fingerprint(fresh.projection.state_dict())
    audit = load_aligned_tower(fresh.encoder, tmp_path, 'infonce', cache)
    assert tensor_fingerprint(fresh.encoder.state_dict()) == done['selected_encoder_sha256']
    assert tensor_fingerprint(fresh.projection.state_dict()) == projection
    assert audit['cache_identity_verified']
    with pytest.raises(ValueError, match='identity mismatch'):
        load_aligned_tower(fresh.encoder, tmp_path, 'infonce', {'a':'different_split'})
    state['trainable']['encoder.0.weight'] += 1
    atomic_save(state, tmp_path/'best.pt')
    with pytest.raises(ValueError, match='fingerprint'):
        load_aligned_tower(fresh.encoder, tmp_path, 'infonce', cache)
