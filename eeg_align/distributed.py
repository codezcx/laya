"""Trainable-gradient synchronization and atomic checkpoint writes."""
import os
from pathlib import Path
import torch
import torch.distributed as dist

@torch.no_grad()
def average_gradients(parameters,world_size):
    parameters=list(parameters)
    if not parameters or any(p.grad is None for p in parameters):raise RuntimeError('Missing active gradient')
    flat=torch.cat([p.grad.reshape(-1) for p in parameters])
    if world_size>1:dist.all_reduce(flat);flat.div_(world_size)
    if not torch.isfinite(flat).all():raise FloatingPointError('Nonfinite synchronized gradient')
    offset=0
    for p in parameters:
        p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()

def atomic_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()
