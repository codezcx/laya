import json
import pytest
import torch
from eeg_align.benchmark_data import BalancedSampler
from eeg_align.matched_control import select_rows, task_sum_loss, sampling_proof
from eeg_align.monitor import update_monitor
from eeg_align.task_options import make_question
from eeg_align.benchmark_data import catalog


def toy_rows():
    return [dict(id=f'{t}-{c}-{s}-{i}',task=t,dataset=d,label=c,subject=str(s))
            for t,d,k in [(0,'BNCI2014001',4),(8,'SEED',3)]
            for c in range(k) for s in range(3) for i in range(5)]


def test_anchor_exposure_and_options_match_with_added_task():
    joint=toy_rows();single=select_rows(joint,['BNCI2014001'])
    a=BalancedSampler(single);b=BalancedSampler(joint)
    assert sampling_proof(single,a,42,600,8,'BNCI2014001') == sampling_proof(joint,b,42,600,8,'BNCI2014001')
    task=catalog()[0]
    for step in [1,99,100,101,600]:
        aa=[single[i] for i in a(single,step,batch=8)]
        bb=[joint[i] for i in b(joint,step,batch=16) if joint[i]['task']==0]
        assert aa==bb
        assert [sum(r['label']==c for r in aa) for c in range(4)]==[2]*4
        assert [make_question(task,task['classes'][r['label']]['class_id'],r['id'],step) for r in aa] == [make_question(task,task['classes'][r['label']]['class_id'],r['id'],step) for r in bb]


def test_task_sum_loss_does_not_dilute_anchor_and_accumulates_microbatches():
    p=torch.tensor(2.,requires_grad=True);q=torch.tensor(3.,requires_grad=True)
    rows=[dict(dataset='MI')]*8+[dict(dataset='SEED')]*8
    weights={'MI':1.,'SEED':1.}
    losses=torch.cat([p.expand(8),q.expand(8)])
    loss=task_sum_loss(losses,rows,8,weights)
    assert loss.item()==5
    loss.backward();assert p.grad.item()==1 and q.grad.item()==1
    p.grad=None;q.grad=None
    for lo,hi in [(0,3),(3,11),(11,16)]:
        losses=torch.cat([p.expand(8),q.expand(8)])
        task_sum_loss(losses[lo:hi],rows[lo:hi],8,weights).backward()
    assert p.grad.item()==1 and q.grad.item()==1


def test_missing_dataset_fails_closed():
    with pytest.raises(ValueError):select_rows(toy_rows(),['missing'])


def test_monitor_rejects_mismatched_anchor_schedule(tmp_path):
    for name,value in [('a','one'),('b','two')]:
        p=tmp_path/name;p.mkdir();(p/'sampling.json').write_text(json.dumps({'step_sha256':[value]}))
    with pytest.raises(RuntimeError,match='Anchor sample'):update_monitor(tmp_path,['a','b'])


def test_monitor_compares_actual_observed_batches(tmp_path):
    for name,value in [('a','one'),('b','two')]:
        p=tmp_path/name;p.mkdir()
        (p/'metrics.jsonl').write_text(json.dumps(dict(step=1,anchor_sample_sha256=value))+'\n')
    with pytest.raises(RuntimeError,match='Observed anchor'):update_monitor(tmp_path,['a','b'])


def test_rng_content_fingerprint_reproducible_and_sensitive():
    from eeg_align.checkpoint import seed_rng,capture_rng,rng_fingerprint,restore_rng
    seed_rng(42);state=capture_rng();before=rng_fingerprint(state)
    torch.rand(5)
    assert rng_fingerprint(capture_rng())!=before
    restore_rng(state)
    assert rng_fingerprint(capture_rng())==before
