"""Trainable-gradient synchronization and atomic checkpoint writes."""
import os
from pathlib import Path
import torch
import torch.distributed as dist


def rank_rows(rows, rank, world):
    """Partition the already sampled global batch, never resample per rank."""
    if world < 1 or not 0 <= rank < world or not rows or len(rows) % world:
        raise ValueError('Global batch must be nonempty and divisible by rank count')
    size = len(rows) // world
    return rows[rank * size:(rank + 1) * size]

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
