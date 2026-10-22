"""Resolve all SEED electrodes using REVE and the supplied SEED template."""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_seed_positions(channels, config_path='configs/seed_positions.json'):
    c=json.loads(Path(config_path).read_text())
    for ds,path in c['template_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=c['template_sha256'][ds]:
            raise ValueError(f'{ds} channel template changed')
    bank=json.loads((Path(c['positions_checkpoint'])/'positions.json').read_text())
    bank={n.upper():xyz for n,xyz in bank.items()}
    bank.update({n.upper():xyz for n,xyz in c['coordinate_overrides'].items()})
    names=[n.strip().upper() for n in channels]
    if len(set(names))!=len(names):raise ValueError('Duplicate EEG channels')
    missing=[n for n in names if n not in bank]
    if missing:raise ValueError(f'Unknown EEG channels: {missing}')
    positions=np.asarray([bank[n] for n in names],dtype=np.float32)
    if positions.shape!=(len(names),3) or not np.isfinite(positions).all():
        raise ValueError('Invalid EEG coordinates')
    return positions
