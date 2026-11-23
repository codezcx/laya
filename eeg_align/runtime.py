"""Current Tower-only LEAF/JEV recipe and content hashing."""
import hashlib,json
from pathlib import Path
CONFIG=json.loads(Path('configs/benchmark_train.json').read_text())


def arm_settings(arm, config=None):
    config = CONFIG if config is None else config
    if arm in config.get('arm_settings', {}):
        return config['arm_settings'][arm]
    raise ValueError(f'Unknown experimental arm: {arm}')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
