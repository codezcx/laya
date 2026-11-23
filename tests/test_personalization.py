import pytest
import torch
from torch import nn

from eeg_align.personalization import (CONFIG, QVDelta, ResidualAdapter, check_support_rows,
    copy_trainable, frozen_hash, install_lora, select_support, separate_session_rows,
    trainable_state, configure_lm, fixed_server_versions, copy_named)
from eeg_align.task_options import make_variant


def rows_2a():
    return [dict(id=f'{session}:{i}', record_id=f'A08{session}', event=i%48,row=i,samples=800,
                 subject='8', split='test', label=i%4)
            for session in ('T','E') for i in range(288)]


def test_sessions_and_nested_shots():
    support, query = separate_session_rows(rows_2a(), 'bcic_2a', 4)
    assert len(support) == len(query) == 288
    previous = set()
    for k in (5,10,20):
        chosen = select_support(support, 4, k, 42)
        ids = {r['id'] for r in chosen}
        assert len(ids) == 4*k and previous <= ids
        assert not ids & {r['id'] for r in query}
        previous = ids
    assert select_support(support,4,5,42) != select_support(support,4,5,43)


def test_duplicate_windows_must_not_be_independent_trials():
    rows = rows_2a()
    rows.append(dict(rows[0], id='another_window'))
    with pytest.raises(ValueError, match='Multiple windows'):
        separate_session_rows(rows, 'bcic_2a', 4)


def test_physio_crossrun_and_short_support_rejected():
    rows = [dict(id=f'{run}:{i}', record_id=f'S089R{run:02d}', event=i,
                 subject='89', split='test', label=(i%2 if run in (4,8,12) else i%2+2))
            for run in (4,6,8,10,12,14) for i in range(14)]
    support, query = separate_session_rows(rows, 'motor_mv_img', 4)
    assert len(support) == 56 and len(query) == 28
    assert len(select_support(support,4,10,42)) == 40
    with pytest.raises(ValueError, match='independent trials'):
        select_support(support,4,20,42)


def test_only_exact_support_can_receive_gradients():
    support, query = separate_session_rows(rows_2a(), 'bcic_2a', 4)
    support = select_support(support,4,5,42)
    case = dict(subject='8', original_split='test', support_ids=[r['id'] for r in support],
                query_ids=[r['id'] for r in query])
    check_support_rows(support, case)
    check_support_rows([support[0]]*16, case)
    with pytest.raises(ValueError, match='authorized support'):
        check_support_rows(query[:1], case)
    with pytest.raises(ValueError, match='authorized support'):
        check_support_rows([dict(support[0],subject='9')], case)
    with pytest.raises(ValueError, match='leakage'):
        check_support_rows(support, dict(case,query_ids=[support[0]['id']]))


def test_residual_identity_update_and_reload():
    torch.manual_seed(42)
    model = ResidualAdapter()
    assert sum(p.numel() for p in model.parameters()) == 33088
    x = torch.randn(8,16,256)
    assert torch.equal(model(x),x)
    initial = trainable_state(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    model(x).square().mean().backward()
    optimizer.step()
    adapted = model(x).detach()
    assert not torch.equal(adapted,x)
    state = trainable_state(model)
    copy_trainable(model,initial)
    assert torch.equal(model(x),x)
    copy_trainable(model,state)
    assert torch.equal(model(x),adapted)


def test_qv_delta_preserves_k_and_original():
    original = nn.Parameter(torch.randn(768,256),requires_grad=False)
    delta = QVDelta(256,8,8)
    assert torch.equal(delta(original),original)
    optimizer = torch.optim.AdamW(delta.parameters(),lr=1e-4)
    delta(original).square().mean().backward()
    optimizer.step()
    current = delta(original)
    assert torch.equal(current[256:512],original[256:512])
    assert not torch.equal(current[:256],original[:256])
    assert not torch.equal(current[512:],original[512:])
    assert original.grad is None
    delta.enabled = False
    assert torch.equal(delta(original),original)


def test_last_layers_only_and_frozen_hash():
    encoder = nn.Module()
    encoder.tower = nn.Module()
    for name in ('mask_transformer','next_transformer'):
        branch = nn.Module()
        branch.layers = nn.ModuleList([nn.TransformerEncoderLayer(256,8,batch_first=True) for _ in range(3)])
        setattr(encoder.tower,name,branch)
    names = install_lora(encoder)
    assert len(names) == 4 and all('.layers.0.' not in n for n in names)
    parameters = [(n,p) for n,p in encoder.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _,p in parameters) == 32768
    assert all('.parametrizations.in_proj_weight.0.' in n for n,_ in parameters)
    before = frozen_hash(encoder)
    with torch.no_grad():
        for _,p in parameters:
            p.add_(0.1)
    assert frozen_hash(encoder) == before


def test_query_prompt_is_label_independent():
    task = dict(instruction='Imagine moving a body part.',question='Which body part?',
                classes=[dict(class_id=str(i),option=str(i)) for i in range(4)])
    prompts = [make_variant(task,i,'sample',0,42,0) for i in range(4)]
    assert len({v['suffix'] for v in prompts}) == 1
    assert all(v['option_class_ids'] == ['0','1','2','3'] for v in prompts)


def test_budget_and_objective_are_matched():
    assert CONFIG['arms'] == ['residual_frozen','residual_lm_lora','lora_frozen','lora_lm_lora']
    assert CONFIG['steps'] == 100 and CONFIG['batch_size'] == 16
    assert CONFIG['seeds'] == [42]
    assert CONFIG['lora_targets'] == ['q','v']
    assert CONFIG['datasets'] == {'bcic_2a':[5,10,20], 'motor_mv_img':[5,10]}
    assert CONFIG['objective']=='jev_ce' and CONFIG['microbatch']==4
    assert CONFIG['learning_rate']==1e-4 and CONFIG['lm_learning_rate']==5e-6


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Linear(4,4)
        self.lora_A=nn.Parameter(torch.randn(2,4)*.1)
        self.lora_B=nn.Parameter(torch.randn(4,2)*.1)

    def forward(self,x):
        return self.base(x)+x@self.lora_A.T@self.lora_B.T


def test_only_language_lora_opened_and_base_unchanged():
    server=nn.Module()
    server.text=TinyLM()
    server.connector=nn.Linear(4,4)
    server.head_weight=nn.Parameter(torch.randn(4,4))
    server.eeg_encoder=nn.Linear(4,4)
    encoder_flags=[p.requires_grad for p in server.eeg_encoder.parameters()]
    parameters=configure_lm(server,True)
    assert set(parameters)=={'text.lora_A','text.lora_B'}
    assert [p.requires_grad for p in server.eeg_encoder.parameters()]==encoder_flags
    fixed=fixed_server_versions(server)
    initial={n:p.detach().clone() for n,p in parameters.items()}
    opt=torch.optim.AdamW(list(parameters.values()),lr=5e-6)
    x=torch.randn(5,4,requires_grad=True)
    server.text(server.connector(x)).square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum()>0
    assert all(p.grad is not None for p in parameters.values())
    assert server.connector.weight.grad is None and server.text.base.weight.grad is None
    opt.step()
    assert fixed_server_versions(server)==fixed
    assert any(not torch.equal(p,initial[n]) for n,p in parameters.items())
    copy_named(parameters,initial)
    assert all(torch.equal(p,initial[n]) for n,p in parameters.items())


def test_frozen_lm_still_passes_input_gradient():
    server=nn.Module()
    server.text=TinyLM()
    assert configure_lm(server,False)=={}
    x=torch.randn(5,4,requires_grad=True)
    server.text(x).square().mean().backward()
    assert x.grad.abs().sum()>0
    assert all(p.grad is None for p in server.parameters())


def test_accumulation_matches_full_ce_gradient():
    torch.manual_seed(11)
    a=nn.Linear(4,4)
    b=nn.Linear(4,4)
    b.load_state_dict(a.state_dict())
    x=torch.randn(16,4)
    y=torch.arange(16)%4
    nn.functional.cross_entropy(a(x),y).backward()
    for start in range(0,16,4):
        (nn.functional.cross_entropy(b(x[start:start+4]),y[start:start+4])/4).backward()
    assert torch.allclose(a.weight.grad,b.weight.grad,atol=1e-7)
    assert torch.allclose(a.bias.grad,b.bias.grad,atol=1e-7)
