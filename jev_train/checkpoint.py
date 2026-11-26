"""Small fine-tuning checkpoints with complete buffers and reproducible RNG state."""
import hashlib
import random
import numpy as np
import torch


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def rng_fingerprint(state):
    """Stable content hash (pickle storage identifiers are not reproducible)."""
    h=hashlib.sha256()
    h.update(repr(state['python']).encode())
    numpy_state=state['numpy']
    h.update(repr((numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:])).encode())
    h.update(tensor_fingerprint({k:state[k] for k in ['torch','cuda'] if state[k] is not None}).encode())
    return h.hexdigest()


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG cannot be restored on CPU')
        torch.cuda.set_rng_state(state['cuda'].cpu())


def capture_model(model):
    return dict(format_version=3,
                trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                # New connector has random frozen bases: save them, not just its LoRA.
                initialized_frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters()
                                    if n.startswith('connector.') and not p.requires_grad},
                buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()})


@torch.no_grad()
def restore_model(model, state):
    if state.get('format_version') != 3 or 'buffers' not in state or 'initialized_frozen' not in state:
        raise ValueError('Checkpoint lacks LoRA connector base or complete buffers')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    fixed={n:p for n,p in model.named_parameters() if n.startswith('connector.') and not p.requires_grad}
    buffers=dict(model.named_buffers())
    if (set(parameters)!=set(state['trainable']) or set(buffers)!=set(state['buffers'])
            or set(fixed)!=set(state['initialized_frozen'])):
        raise ValueError('Checkpoint parameter/buffer names differ from model')
    # Validate all shapes before modifying model state.
    pairs = [(parameters,state['trainable']), (fixed,state['initialized_frozen']), (buffers,state['buffers'])]
    for actual,saved in pairs:
        for name,target in actual.items():
            if target.shape!=saved[name].shape or target.dtype!=saved[name].dtype:
                raise ValueError(f'Checkpoint shape/dtype mismatch: {name}')
    for actual,saved in pairs:
        for name,target in actual.items():
            target.copy_(saved[name])


def tensor_fingerprint(tensors):
    h=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
