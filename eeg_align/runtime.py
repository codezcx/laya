"""Current BSR/JEV recipe and content hashing shared by training/queue."""
import hashlib,json
from pathlib import Path
CONFIG=json.loads(Path('configs/benchmark_train.json').read_text())


def arm_settings(arm, config=None):
    config = CONFIG if config is None else config
    if arm in config.get('arm_settings', {}):
        return config['arm_settings'][arm]
    if arm in ('mean_pool', 'cls', 'cross_attention', 'qformer'):
        return dict(encoder='bsr', encoder_trainable=False, readout=arm)
    raise ValueError(f'Unknown experimental arm: {arm}')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
