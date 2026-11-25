"""Read-only provenance checks for the completed source-trained initializations."""
import json
import torch
from jev_train.checkpoint import tensor_fingerprint
from .common import CONFIG, ROOT, digest
from .modules import UserParameters


def common_state():
    # Same CPU constructor as make_user before .cuda(); do not consume caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CONFIG['initialization_seed'])
        return UserParameters(CONFIG).state_dict()


def audit_initializations():
    root = ROOT / CONFIG['initialization_reference']
    previous = json.loads((root / 'plan.json').read_text())
    queue = json.loads((root / 'queue.json').read_text())
    if queue['state'] != 'complete' or queue['complete_jobs'] != 4:
        raise RuntimeError('Initialization reference is not a completed experiment')
    fixed = ('parent', 'parent_sha256', 'seed', 'steps', 'support_total', 'batch_size',
             'microbatch', 'eval_batch', 'encode_batch', 'learning_rate', 'weight_decay',
             'clip_grad_norm', 'connector_lora_rank', 'connector_lora_scale', 'cache_root',
             'split_root', 'catalog', 'hmc_support_fraction', 'hmc_gap_epochs')
    for name in fixed:
        if CONFIG[name] != previous['config'][name]:
            raise RuntimeError(f'Unapproved hyperparameter/input change: {name}')
    if CONFIG['initialization_seed'] != previous['config']['seed']:
        raise RuntimeError('Common seed differs from source training initial seed')
    start = common_state()
    starting_sha = tensor_fingerprint(start)
    records = {}
    for dataset in CONFIG['datasets']:
        records[dataset] = {}
        for arm in CONFIG['arms']:
            if arm == 'ordinary_lora':
                records[dataset][arm] = dict(path=None, starting_sha=starting_sha,
                    final_sha=starting_sha, seed=CONFIG['initialization_seed'], source_episodes=0)
                continue
            directory = root / 'jobs' / f'{dataset}_{arm}'
            complete = json.loads((directory / 'offline_complete.json').read_text())
            if complete['episodes'] != 60 or complete['starting_sha'] != starting_sha:
                raise RuntimeError('Source-training budget/common random initialization changed')
            if complete['job']['dataset'] != dataset or complete['job']['arm'] != arm:
                raise RuntimeError('Source initialization task/arm mismatch')
            if complete['stamp']['parent_sha256'] != CONFIG['parent_sha256']:
                raise RuntimeError('Source initialization parent differs')
            if complete['stamp']['source_hashes'] != previous['source_hashes']:
                raise RuntimeError('Reference source code provenance differs')
            path = directory / 'initialization.pt'
            state = torch.load(path, weights_only=True, map_location='cpu')
            if set(state) != set(start) or any(state[k].shape != start[k].shape or
                    state[k].dtype != start[k].dtype or not state[k].isfinite().all() for k in start):
                raise RuntimeError('Initialization tensor schema changed')
            if tensor_fingerprint(state) != complete['final_sha']:
                raise RuntimeError('Source initialization fingerprint changed')
            records[dataset][arm] = dict(path=str(path.relative_to(ROOT)), file_sha256=digest(path),
                starting_sha=starting_sha, final_sha=complete['final_sha'],
                seed=CONFIG['initialization_seed'], source_episodes=complete['episodes'])
    return records


def load_initialization(record):
    if record['path'] is None:
        state = common_state()
    else:
        path = ROOT / record['path']
        if digest(path) != record['file_sha256']:
            raise RuntimeError('Initialization file changed since planning')
        state = torch.load(path, weights_only=True, map_location='cpu')
    if tensor_fingerprint(state) != record['final_sha']:
        raise RuntimeError('Initialization parameter fingerprint mismatch')
    return state
