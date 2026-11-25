"""Same zero-delta start in both Tower-freezing controls."""
import json
import torch
from jev_train.checkpoint import tensor_fingerprint
from .common import CONFIG, ROOT
from .modules import UserParameters


def common_state():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CONFIG['initialization_seed'])
        return UserParameters(CONFIG).state_dict()


def audit_initializations():
    root = ROOT / CONFIG['matched_reference']
    previous = json.loads((root / 'plan.json').read_text())
    if json.loads((root / 'queue.json').read_text())['state'] != 'complete':
        raise RuntimeError('Matched control reference is incomplete')
    fixed = ('parent', 'parent_sha256', 'seed', 'steps', 'support_total', 'batch_size',
             'microbatch', 'eval_batch', 'encode_batch', 'learning_rate', 'weight_decay',
             'clip_grad_norm', 'connector_lora_rank', 'connector_lora_scale', 'cache_root',
             'split_root', 'catalog', 'hmc_support_fraction', 'hmc_gap_epochs',
             'initialization_seed', 'eval_steps')
    for name in fixed:
        if CONFIG[name] != previous['config'][name]:
            raise RuntimeError(f'Unapproved matched-control change: {name}')
    if CONFIG['tower_learning_rate'] != CONFIG['learning_rate']:
        raise RuntimeError('This pilot uses the same declared Tower/connector learning rate')
    fingerprint = tensor_fingerprint(common_state())
    if fingerprint != '88da1153c2e270122ff7e9af1bb9db58e5824a9fc415426022e66af5f8112542':
        raise RuntimeError('Historical random-A/zero-B initialization changed')
    return {dataset: {arm: dict(starting_sha=fingerprint, final_sha=fingerprint,
                               seed=CONFIG['initialization_seed'], source_episodes=0)
                      for arm in CONFIG['arms']} for dataset in CONFIG['datasets']}


def load_initialization(record):
    state = common_state()
    if tensor_fingerprint(state) != record['final_sha']:
        raise RuntimeError('Initialization parameter fingerprint mismatch')
    return state
