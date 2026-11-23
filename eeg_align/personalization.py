"""Matched support-only signal adaptation with frozen versus LoRA-updated JEV."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .checkpoint import tensor_fingerprint
from .contrastive_objectives import class_scores, per_sample_loss, positive_mask
from .contrastive_protocol import load_text_bank
from .online_leaf import OnlineLEAFTower, upstream_files
from .runtime import CONFIG as DATA, digest

CONFIG_PATH = Path('configs/personalization.json')
CONFIG = json.loads(CONFIG_PATH.read_text())
ROOT = Path('artifacts/benchmark') / CONFIG['run_name']


def source_identity():
    files = list(Path('eeg_align').glob('*.py'))
    files += [CONFIG_PATH, Path('scripts/run_personalization.py')]
    files += [Path(p) for p in ('configs/benchmark_train.json', 'configs/benchmark_tasks.json',
                               'configs/contrastive_train.json', 'scripts/prepare_multitask_signals.py')]
    files += upstream_files(DATA)
    return {str(p): digest(p) for p in sorted(set(files))}


def ids_hash(rows):
    return hashlib.sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest()


def trial_key(row):
    # 2a cache concatenates six 48-trial runs, resetting event/start each run.
    run = int(row['row']) // 48 if re.fullmatch(r'A\d{2}[TE]', row['record_id']) else 0
    return (row['record_id'], run, int(row['event']))


def separate_session_rows(rows, name, classes):
    """MI caches must contain exactly one independent window per event."""
    if len({r['subject'] for r in rows}) != 1 or len({r['split'] for r in rows}) != 1:
        raise ValueError('Expected exactly one subject and original split')
    if len({trial_key(r) for r in rows}) != len(rows):
        raise ValueError('Multiple windows from a trial: group them before few-shot sampling')
    if name == 'bcic_2a':
        for record in {r['record_id'] for r in rows}:
            part = sorted((r for r in rows if r['record_id'] == record), key=lambda r:int(r['row']))
            if (len(part) != 288 or [int(r['row']) for r in part] != list(range(288))
                    or any(int(r['event']) != i % 48 or int(r['samples']) != 800 for i,r in enumerate(part))):
                raise ValueError('2a cache must have six complete 48-trial runs, one four-second window per event')
    support, query = [], []
    for r in rows:
        record = r['record_id']
        if name == 'bcic_2a':
            if not re.fullmatch(r'A\d{2}[TE]', record):
                raise ValueError('Unknown 2a session')
            target = support if record.endswith('T') else query
        elif name == 'motor_mv_img':
            match = re.fullmatch(r'S\d{3}R(\d{2})', record)
            if not match or int(match[1]) not in (4, 6, 8, 10, 12, 14):
                raise ValueError('Unknown PhysioMI imagery run')
            target = support if int(match[1]) <= 10 else query
        else:
            raise ValueError('Unapproved dataset')
        target.append(r)
    for partition in (support, query):
        if {int(r['label']) for r in partition} != set(range(classes)):
            raise ValueError('Missing class in calibration or query session')
    if set(r['record_id'] for r in support) & set(r['record_id'] for r in query):
        raise ValueError('Record leakage')
    return sorted(support, key=trial_key), sorted(query, key=trial_key)


def select_support(pool, classes, shots, seed):
    # The same per-class permutation makes 5-shot a subset of 10/20-shot.
    selected = []
    for label in range(classes):
        choices = sorted([r for r in pool if int(r['label']) == label], key=trial_key)
        if len(choices) < shots:
            raise ValueError(f'Only {len(choices)} independent trials for class {label}; need {shots}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, label, 711]))
        selected.extend(choices[i] for i in rng.permutation(len(choices))[:shots])
    return selected


def make_manifest(store, smoke=False):
    split = 'valid' if smoke else 'test'
    cases, cohorts = [], []
    for tid, task in enumerate(store.tasks):
        name = task['task_id']
        if name not in CONFIG['datasets']:
            continue
        grouped = defaultdict(list)
        for r in store.rows[tid]:
            if r['split'] == split:
                grouped[r['subject']].append(r)
        for subject, rows in sorted(grouped.items(), key=lambda kv: int(kv[0])):
            if smoke and subject != min(grouped, key=int):
                continue
            pool, query = separate_session_rows(rows, name, len(task['classes']))
            cohorts.append(dict(dataset=name, subject=subject, original_split=split,
                support_pool_ids=[r['id'] for r in pool], query_ids=[r['id'] for r in query],
                support_records=sorted({r['record_id'] for r in pool}),
                query_records=sorted({r['record_id'] for r in query}), query_sha256=ids_hash(query)))
            for seed in ([CONFIG['seed']] if smoke else CONFIG['seeds']):
                for shots in ([5] if smoke else CONFIG['datasets'][name]):
                    support = select_support(pool, len(task['classes']), shots, seed)
                    for arm in CONFIG['arms']:
                        cases.append(dict(key=f'{name}_s{subject}_{arm}_k{shots}_seed{seed}',
                            dataset=name, task=tid, subject=subject, arm=arm, shots_per_class=shots,
                            **CONFIG['arm_settings'][arm],
                            seed=seed, original_split=split, support_ids=[r['id'] for r in support],
                            query_ids=[r['id'] for r in query], support_sha256=ids_hash(support),
                            query_sha256=ids_hash(query)))
    for case in cases:
        case['worker'] = CONFIG['arms'].index(case['arm'])
    return dict(config=CONFIG, smoke=smoke, original_split=split, cohorts=cohorts, cases=cases,
                cache_indexes={store.tasks[t]['task_id']: m['index_sha256'] for t, m in store.meta.items()},
                query_used_for_selection=False, original_splits_modified=False)


def check_support_rows(rows, case):
    allowed = set(case['support_ids'])
    if not rows or any(r['id'] not in allowed or r['subject'] != case['subject']
                       or r['split'] != case['original_split'] for r in rows):
        raise ValueError('Personal gradients may use only this case\'s authorized support trials')
    if allowed & set(case['query_ids']):
        raise ValueError('Support/query leakage')


class ResidualAdapter(nn.Module):
    def __init__(self, width=256, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class QVDelta(nn.Module):
    """Packed MHA Q/V LoRA; K, output projection and original Q/V are frozen."""
    def __init__(self, width, rank, alpha):
        super().__init__()
        self.enabled = True
        self.scale = alpha / rank
        self.q_A = nn.Parameter(torch.empty(rank, width))
        self.q_B = nn.Parameter(torch.zeros(width, rank))
        self.v_A = nn.Parameter(torch.empty(rank, width))
        self.v_B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))

    def forward(self, original):
        if not self.enabled:
            return original
        q = self.q_B @ self.q_A
        v = self.v_B @ self.v_A
        return original + self.scale * torch.cat([q, torch.zeros_like(q), v], dim=0)


def install_lora(encoder, config=CONFIG):
    encoder.requires_grad_(False)
    names = []
    for branch in ('mask_transformer', 'next_transformer'):
        layers = getattr(encoder.tower, branch).layers
        for i in range(len(layers)-config['lora_last_layers_per_branch'], len(layers)):
            attn = layers[i].self_attn
            if not isinstance(attn, nn.MultiheadAttention) or not attn._qkv_same_embed_dim:
                raise ValueError('Unexpected LEAF attention layout')
            delta = QVDelta(attn.embed_dim, config['lora_rank'], config['lora_alpha'])
            delta.to(device=attn.in_proj_weight.device, dtype=attn.in_proj_weight.dtype)
            parametrize.register_parametrization(attn, 'in_proj_weight', delta)
            names.append(f'tower.{branch}.layers.{i}.self_attn.in_proj_weight')
    return names


def load_parent():
    directory = Path(CONFIG['parent'])
    done = json.loads((directory/'complete.json').read_text())
    saved = torch.load(directory/'best.pt', map_location='cpu', mmap=True, weights_only=False)
    if (done['smoke'] or done['test_evaluations'] or saved['step'] != done['selected_step']
            or saved['identity']['config'] != DATA):
        raise ValueError('Invalid completed parent checkpoint')
    return saved


def load_encoder(saved, device='cuda'):
    encoder = OnlineLEAFTower(DATA)
    source = {k.removeprefix('eeg_encoder.'): v for group in ('trainable', 'buffers')
              for k, v in saved[group].items() if k.startswith('eeg_encoder.')}
    encoder.load_state_dict(source, strict=True)
    encoder.requires_grad_(False)
    return encoder.to(device).eval()


def projection_module():
    return nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 768), nn.GELU(), nn.Linear(768, 768))


def pooled(tokens, valid):
    return (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)


class TextReadout(nn.Module):
    def __init__(self, projection, scale):
        super().__init__()
        self.projection = projection
        self.register_buffer('log_scale', scale)
        self.bank, vectors = load_text_bank()
        self.register_buffer('vectors', torch.from_numpy(vectors))

    def forward(self, tokens, valid, name, labels=None):
        z = F.normalize(self.projection(pooled(tokens, valid)).float(), dim=-1)
        bank = self.bank['banks'][name]
        text_labels = torch.tensor(bank['class_labels'], device=z.device)
        cosine = z @ self.vectors[bank['text_indices']].T
        scores = class_scores(cosine, text_labels, len(bank['semantic_ids']))
        loss = None if labels is None else per_sample_loss(cosine * self.log_scale.exp(),
                        positive_mask(labels, text_labels), 'infonce').mean()
        return scores, loss


def trainable_state(module):
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters() if p.requires_grad}


def frozen_hash(module):
    return tensor_fingerprint({**{n: p for n, p in module.named_parameters() if not p.requires_grad},
                               **dict(module.named_buffers())})


@torch.no_grad()
def copy_trainable(module, state):
    parameters = {n: p for n, p in module.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('Personal parameter keys changed')
    for n, p in parameters.items():
        p.copy_(state[n])


def is_lm_lora(name):
    return name.startswith('text.') and name.endswith(('lora_A', 'lora_B'))


def configure_lm(server, enabled):
    """Signal trainability is managed separately; connector/head never update."""
    selected = {}
    for name, parameter in server.named_parameters():
        if name.startswith('eeg_encoder.'):
            continue
        parameter.requires_grad_(enabled and is_lm_lora(name))
        if parameter.requires_grad:
            selected[name] = parameter
    return selected


def fixed_server_versions(server):
    return {n:p._version for n,p in server.named_parameters()
            if not n.startswith('eeg_encoder.') and not p.requires_grad}


@torch.no_grad()
def copy_named(parameters, state):
    if parameters.keys() != state.keys():
        raise ValueError('LM adapter parameter keys changed')
    for n,p in parameters.items():
        p.copy_(state[n])
