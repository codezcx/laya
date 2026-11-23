"""Current approach: support-set amortized FiLM with a frozen JEV server."""
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
    run = int(row['row']) // 48 if re.fullmatch(r'A\d{2}[TE]', row['record_id']) else 0
    return (row['record_id'], run, int(row['event']))


def separate_session_rows(rows, name, classes):
    if len({r['subject'] for r in rows}) != 1 or len({r['split'] for r in rows}) != 1:
        raise ValueError('Expected one subject and original split')
    if len({trial_key(r) for r in rows}) != len(rows):
        raise ValueError('Multiple windows from a trial')
    if name == 'bcic_2a':
        for record in {r['record_id'] for r in rows}:
            part = sorted((r for r in rows if r['record_id'] == record), key=lambda r:int(r['row']))
            if (len(part) != 288 or [int(r['row']) for r in part] != list(range(288))
                    or any(int(r['event']) != i % 48 or int(r['samples']) != 800 for i,r in enumerate(part))):
                raise ValueError('2a must have six complete 48-trial runs')
    support, query = [], []
    for r in rows:
        record = r['record_id']
        if name == 'bcic_2a':
            if not re.fullmatch(r'A\d{2}[TE]', record):
                raise ValueError('Unknown 2a session')
            target = support if record.endswith('T') else query
        elif name == 'motor_mv_img':
            match = re.fullmatch(r'S\d{3}R(\d{2})', record)
            if not match or int(match[1]) not in (4,6,8,10,12,14):
                raise ValueError('Unknown PhysioMI imagery run')
            target = support if int(match[1]) <= 10 else query
        else:
            raise ValueError('Unapproved dataset')
        target.append(r)
    for partition in (support, query):
        if {int(r['label']) for r in partition} != set(range(classes)):
            raise ValueError('Missing class in support/query')
    if {r['record_id'] for r in support} & {r['record_id'] for r in query}:
        raise ValueError('Record leakage')
    return sorted(support, key=trial_key), sorted(query, key=trial_key)


def select_support(pool, classes, shots, seed):
    selected = []
    for label in range(classes):
        choices = sorted([r for r in pool if int(r['label']) == label], key=trial_key)
        if len(choices) < shots:
            raise ValueError(f'Only {len(choices)} independent trials for class {label}; need {shots}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, label, 711]))
        selected.extend(choices[i] for i in rng.permutation(len(choices))[:shots])
    return selected


def make_manifest(store, config=CONFIG):
    cohorts, excluded = [], {}
    for tid, task in enumerate(store.tasks):
        name = task['task_id']
        if name not in config['datasets']:
            continue
        grouped = defaultdict(list)
        for row in store.rows[tid]:
            grouped[(row['split'], row['subject'])].append(row)
        excluded[name] = len(store.excluded[tid])
        for (split, subject), rows in sorted(grouped.items(), key=lambda kv:(kv[0][0], int(kv[0][1]))):
            if split != 'train' and subject not in config[{'valid':'validation_subjects','test':'test_subjects'}[split]].get(name, []):
                continue
            support, query = separate_session_rows(rows, name, len(task['classes']))
            if split == 'train':
                allowed = {r['id'] for i,r in enumerate(store.rows[tid]) if i not in store.excluded[tid]}
                support = [r for r in support if r['id'] in allowed]
                query = [r for r in query if r['id'] in allowed]
            select_support(support, len(task['classes']), config['shots_per_class'], config['seed'])
            if any(sum(int(r['label']) == c for r in query) < config['global_query_batch']//len(task['classes'])
                   for c in range(len(task['classes']))):
                raise ValueError('Insufficient independent query trials')
            cohorts.append(dict(key=f'{name}_s{subject}',dataset=name,task=tid,subject=subject,split=split,
                classes=len(task['classes']),support_pool=support,query=query,
                support=select_support(support,len(task['classes']),config['shots_per_class'],config['seed'])))
    for split, key in [('valid','validation_subjects'),('test','test_subjects')]:
        expected = {(n,s) for n,ss in config[key].items() for s in ss}
        if {(c['dataset'],c['subject']) for c in cohorts if c['split']==split} != expected:
            raise ValueError('Requested held-out subjects not present in original split')
    train = {c['key']:c for c in cohorts if c['split']=='train'}
    episodes = []
    # Equal dataset sampling, round-robin randomized source subjects, no query replacement.
    for step in range(config['steps']):
        cycle, position = divmod(step,len(config['datasets']))
        order = np.random.default_rng(np.random.SeedSequence([config['seed'],cycle,990])).permutation(config['datasets'])
        name = str(order[position])
        choices = sorted([c for c in train.values() if c['dataset']==name],key=lambda c:int(c['subject']))
        epoch, slot = divmod(cycle,len(choices))
        perm = np.random.default_rng(np.random.SeedSequence([config['seed'],config['datasets'].index(name),epoch,991])).permutation(len(choices))
        cohort = choices[int(perm[slot])]
        seed = config['seed'] + 1009*(step+1)
        support = select_support(cohort['support_pool'],cohort['classes'],config['shots_per_class'],seed)
        query = select_support(cohort['query'],cohort['classes'],config['global_query_batch']//cohort['classes'],seed+1)
        episode = dict(step=step+1,key=cohort['key'],support_ids=[r['id'] for r in support],query_ids=[r['id'] for r in query])
        assert_episode(cohort, episode, training=True)
        episodes.append(episode)
    return dict(config=config,cohorts=cohorts,episodes=episodes,qc_excluded=excluded,
        cache_indexes={store.tasks[t]['task_id']:m['index_sha256'] for t,m in store.meta.items()},
        test_used_for_selection=False,original_splits_modified=False)


def assert_episode(cohort, episode, training=False):
    if training and cohort['split'] != 'train':
        raise ValueError('Only original train subjects may receive gradients')
    support, query = episode['support_ids'],episode['query_ids']
    if (len(set(support)) != len(support) or len(set(query)) != len(query)
            or set(support) & set(query)):
        raise ValueError('Duplicate trials or support/query leakage')
    if not set(support) <= {r['id'] for r in cohort['support_pool']} or not set(query) <= {r['id'] for r in cohort['query']}:
        raise ValueError('Episode crossed subject/session boundary')


class AmortizedFiLM(nn.Module):
    """DeepSets G + parameter generator H + tokenwise residual FiLM adapter.

    The fixed semantic vectors encode support labels, never query labels. No
    cosine classifier, class prototypes, task-specific heads or subject-ID lookup.
    """
    def __init__(self, width=256, text_width=768, label_width=64, context_width=128, bottleneck=64):
        super().__init__()
        self.label_projection = nn.Linear(text_width,label_width)
        self.element = nn.Sequential(nn.Linear(width+label_width,context_width),nn.GELU(),
                                     nn.Linear(context_width,context_width),nn.GELU())
        self.context = nn.Sequential(nn.Linear(context_width,context_width),nn.GELU())
        self.condition = nn.Sequential(nn.Linear(context_width,context_width),nn.GELU(),
                                       nn.Linear(context_width,2*bottleneck))
        self.down = nn.Linear(width,bottleneck)
        self.up = nn.Linear(bottleneck,width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def summarize(self, support_tokens, label_vectors):
        if support_tokens.ndim != 3 or len(support_tokens)==0 or len(label_vectors)!=len(support_tokens):
            raise ValueError('Nonempty support tokens and matched semantic labels required')
        z = support_tokens.mean(1)
        r = self.element(torch.cat([z,self.label_projection(label_vectors)],dim=-1))
        return self.context(r.mean(0))

    def forward(self, tokens, context):
        gamma,beta = self.condition(context).chunk(2,dim=-1)
        h = (1+gamma)*self.down(tokens)+beta
        return tokens+self.up(F.gelu(h))


def load_parent():
    directory = Path(CONFIG['parent'])
    done = json.loads((directory/'complete.json').read_text())
    saved = torch.load(directory/'best.pt',map_location='cpu',mmap=True,weights_only=False)
    if (done['smoke'] or done['test_evaluations'] or saved['step'] != done['selected_step']
            or saved['identity']['config'] != DATA):
        raise ValueError('Invalid completed parent checkpoint')
    return saved


def load_encoder(saved,device='cuda'):
    encoder = OnlineLEAFTower(DATA)
    source = {k.removeprefix('eeg_encoder.'):v for group in ('trainable','buffers')
              for k,v in saved[group].items() if k.startswith('eeg_encoder.')}
    encoder.load_state_dict(source,strict=True)
    return encoder.requires_grad_(False).to(device).eval()


def fixed_server_versions(server):
    return {n:p._version for n,p in server.named_parameters() if not n.startswith('eeg_encoder.')}


def module_hash(module):
    return tensor_fingerprint(module.state_dict())
