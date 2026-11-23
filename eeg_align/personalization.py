"""Independent per-user residual adapters, supervised through a fully frozen JEV."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import tensor_fingerprint
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
    if not smoke:
        observed = {name: sorted(int(c['subject']) for c in cohorts if c['dataset'] == name)
                    for name in CONFIG['datasets']}
        if observed != CONFIG['held_out_subjects']:
            raise ValueError(f'Unexpected held-out subject list: {observed}')
    for index, case in enumerate(cases):
        case['worker'] = index % len(CONFIG['gpus'])
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


def fixed_server_versions(server):
    return {n: p._version for n, p in server.named_parameters()}
