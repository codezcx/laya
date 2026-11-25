"""Small, testable first-order meta-gradient and matched exposure schedule."""
import copy
import numpy as np
import torch


def clone_initialization(initial):
    clone = copy.deepcopy(initial)
    for p in clone.parameters():
        p.grad = None
    return clone


def first_order_gradients(initial, adapted):
    """Copy dL_Q/dphi_adapted to phi0, omitting the inner-update Jacobian."""
    left, right = dict(initial.named_parameters()), dict(adapted.named_parameters())
    if left.keys() != right.keys():
        raise RuntimeError('Meta initialization/inner parameter names differ')
    for name, p in left.items():
        q = right[name]
        if p is q or p.shape != q.shape or q.grad is None or not q.grad.isfinite().all():
            raise RuntimeError('Invalid first-order meta-gradient')
        p.grad = q.grad.detach().clone()


def episode_schedule(sources, config):
    result = []
    for i in range(config['offline']['episodes']):
        cycle, offset = divmod(i, len(sources))
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], cycle, 837]))
        case = sources[int(rng.permutation(len(sources))[offset])]
        batches = []
        for j in range(config['offline']['inner_steps'] + 1):
            part = 'support' if j < config['offline']['inner_steps'] else 'query'
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'], i, j, 839]))
            selected = rng.integers(len(case[part + '_ids']), size=config['batch_size']).tolist()
            batches.append(dict(part=part, indices=selected, ids=[case[part + '_ids'][k] for k in selected],
                prompt_step=i * (config['offline']['inner_steps'] + 1) + j + 1))
        result.append(dict(episode=i + 1, source=case['key'], batches=batches))
    return result
