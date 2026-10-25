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


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG cannot be restored on CPU')
        torch.cuda.set_rng_state(state['cuda'].cpu())


def capture_model(model):
    return dict(format_version=2,
                trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()})


@torch.no_grad()
def restore_model(model, state):
    if state.get('format_version') != 2 or 'buffers' not in state:
        raise ValueError('Checkpoint lacks complete buffers; explicit warmup recovery is required')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    buffers=dict(model.named_buffers())
    if set(parameters)!=set(state['trainable']) or set(buffers)!=set(state['buffers']):
        raise ValueError('Checkpoint parameter/buffer names differ from model')
    # Validate all shapes before modifying model state.
    for actual,saved in [(parameters,state['trainable']),(buffers,state['buffers'])]:
        for name,target in actual.items():
            if target.shape!=saved[name].shape or target.dtype!=saved[name].dtype:
                raise ValueError(f'Checkpoint shape/dtype mismatch: {name}')
    for actual,saved in [(parameters,state['trainable']),(buffers,state['buffers'])]:
        for name,target in actual.items():
            target.copy_(saved[name])


@torch.no_grad()
def recover_frozen_warmup(model, state, freeze_steps):
    """Only old checkpoints whose encoder never changed can recover release buffers."""
    if state.get('format_version') == 2 or not 0 < state['step'] <= freeze_steps:
        raise ValueError('Legacy recovery is limited to the frozen-encoder warmup')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(state['trainable']):
        raise ValueError('Legacy parameter names differ')
    checked=0
    for name,p in parameters.items():
        saved=state['trainable'][name]
        if p.shape!=saved.shape or p.dtype!=saved.dtype:
            raise ValueError(f'Legacy shape/dtype mismatch: {name}')
        if name.startswith('eeg_encoder.'):
            if not torch.equal(p.detach().cpu(),saved):
                raise ValueError(f'Encoder changed; original buffers cannot be recovered: {name}')
            checked+=1
    if not checked:
        raise ValueError('No encoder parameters were checked')
    for name,p in parameters.items():
        p.copy_(state['trainable'][name])
    return dict(legacy_step=state['step'],unchanged_encoder_tensors=checked,
                buffers_source='released model; encoder never unfrozen',
                rng_source='explicit matched branch seed; old RNG was not saved')


def tensor_fingerprint(tensors):
    h=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
