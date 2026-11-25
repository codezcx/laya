import copy
import json
import os
import subprocess
import sys

import pytest
import torch

from scripts import queue_personal_rank as rank
from personal.common import CONFIG, identity
from personal.modules import UserParameters, install_tower_lora, tower_lora_parameters
from jev_train.checkpoint import tensor_fingerprint


@pytest.mark.parametrize('stage', ['connector16', 'tower16'])
def test_rank_plan_reuses_all_people_samples_and_batches(stage):
    baseline = json.loads((rank.REFERENCE/'plan.json').read_text())
    plan = rank.build_plan(stage)
    assert plan['cases'] == baseline['cases']
    assert plan['inventory'] == baseline['inventory']
    assert plan['source_array_stats'] == baseline['source_array_stats']
    assert len(plan['episodes']) == 21
    assert [len(j['episodes']) for j in plan['jobs']] == [6,5,5,5]
    assert len({e['key'] for e in plan['episodes']}) == 21
    assert sorted(k for j in plan['jobs'] for k in j['episodes']) == sorted(e['key'] for e in plan['episodes'])
    for case in plan['cases']:
        key = case['key']
        assert plan['support_sets'][key] == {'100': baseline['support_sets'][key]['100']}
        assert plan['schedules'][f'{key}_n100'] == baseline['schedules'][f'{key}_n100']
    changed = {k for k in CONFIG if CONFIG[k] != plan['config'][k]}
    expected_rank = 'connector_lora_rank' if stage == 'connector16' else 'tower_lora_rank'
    assert changed == {'run_name','output_root','authorization','arms','dose_arms','support_sizes',expected_rank}
    assert CONFIG['connector_lora_rank'] == CONFIG['tower_lora_rank'] == 4
    assert identity() == baseline['source_hashes']


@pytest.mark.parametrize('stage,count', [('connector16',249856),('tower16',62464)])
def test_rank_connector_init_identity_and_gradient(stage,count):
    config = rank.recipe(stage); arm = rank.SPEC['stages'][stage]['arm']
    user = UserParameters(config,arm)
    assert sum(p.numel() for p in user.parameters()) == count
    assert tensor_fingerprint(user.state_dict()) == rank.build_plan(stage)['initializations'][arm]['final_sha']
    a = user.first(torch.randn(2,3,256))
    b = user.delta(torch.randn(2,3,5120))
    assert not a.count_nonzero() and not b.count_nonzero()
    (a.sum()+b.sum()).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in user.parameters())
    assert user.first.b.grad.abs().sum() > 0 and user.delta.b.grad.abs().sum() > 0
    if stage == 'tower16':
        old = UserParameters(CONFIG,'tower_both')
        assert tensor_fingerprint(user.state_dict()) == tensor_fingerprint(old.state_dict())


@pytest.mark.parametrize('stage', ['connector16', 'tower16'])
def test_worker_bootstrap_binds_effective_config_before_import(stage):
    code = f'''
import sys
from scripts import queue_personal_rank as r
r.configure_worker({stage!r})
from personal import worker, model, initialization
from personal.common import identity
assert worker.CONFIG == model.CONFIG == r.recipe({stage!r})
assert worker.OUT == r.OUT/{stage!r}
assert identity() == worker.identity() == r.identity()
arm=r.SPEC['stages'][{stage!r}]['arm']
from personal.modules import UserParameters
from jev_train.checkpoint import tensor_fingerprint
assert tensor_fingerprint(initialization.common_state(arm)) == tensor_fingerprint(UserParameters(worker.CONFIG,arm).state_dict())
try: r.configure_worker({stage!r})
except RuntimeError: pass
else: raise AssertionError('late configuration should fail')
print('bootstrap passed')
'''
    proc = subprocess.run([sys.executable,'-c',code],cwd=rank.ROOT,text=True,capture_output=True,timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert 'bootstrap passed' in proc.stdout


def test_pairing_requires_every_user_and_rejects_changed_membership():
    cases = [dict(key='a'),dict(key='b')]
    old,new = {},{}
    for i,case in enumerate(cases):
        base = dict(case=dict(case,support_ids=['s'],query_ids=['q']),
            baseline=dict(balanced_accuracy=.4),
            support=dict(accuracy=.8),query=dict(balanced_accuracy=.5,ce=1.))
        old[case['key'],100] = copy.deepcopy(base)
        new[case['key'],100] = copy.deepcopy(base)
        new[case['key'],100]['query']['balanced_accuracy'] = .7 if i == 0 else .4
    group = rank.paired_group(new,old,cases,100)
    assert group['delta_pp'] == pytest.approx(5.) and group['negative_users'] == 1
    assert rank.paired_group({('a',100):new['a',100]},old,cases,100) is None
    new['b',100]['case']['query_ids'] = ['different']
    with pytest.raises(RuntimeError,match='changed samples'):
        rank.paired_group(new,old,cases,100)


def test_wait_does_not_start_on_failed_or_running_predecessor():
    assert rank.predecessor_ready(dict(state='personal_workers')) is False
    for state in ('failed','stopped_by_user'):
        with pytest.raises(RuntimeError,match='will not start'):
            rank.predecessor_ready(dict(state=state))
    assert rank.process_token(os.getpid()) is not None
    assert rank.process_token(999999999) is None


def test_real_tower_rank16_count_and_base_freeze():
    from jev_train.online_leaf import OnlineLEAFTower
    from personal.common import PARENT_CONFIG
    tower = OnlineLEAFTower(PARENT_CONFIG).eval().requires_grad_(False)
    records = install_tower_lora(tower,16,1.,42)
    parameters = tower_lora_parameters(tower)
    assert len(records) == 98
    assert sum(p.numel() for p in parameters.values()) == 1601536
    assert all(not p.requires_grad for n,p in tower.named_parameters() if n not in parameters)
    assert all(p.requires_grad for p in parameters.values())
    x = torch.randn(1,65,800)
    tower(x).square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in parameters.values())
    assert all(p.grad is None for n,p in tower.named_parameters() if n not in parameters)
