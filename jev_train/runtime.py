"""MedJev integration; unchanged 13-dataset membership and inherited training budget."""
import json
import struct
from pathlib import Path

from leaf_repro.common import sha256 as digest, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/jev13_train.json').read_text())


def arm_settings(arm):
    return CONFIG['arm_settings'][arm]


def backbone_files():
    root = Path(CONFIG['backbone_path'])
    pending = list(root.glob('*.aria2')) + list(root.glob('*.incomplete'))
    if pending:
        raise FileNotFoundError('Qwen download still in progress: ' + ', '.join(p.name for p in pending))
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    paths = [root / name for name in sorted(set(index['weight_map'].values()))]
    paths += [root / name for name in ('config.json', 'model.safetensors.index.json',
        'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')]
    if any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise FileNotFoundError('Qwen model files missing or empty')
    for path in paths:
        if path.suffix != '.safetensors':
            continue
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise FileNotFoundError(f'Download header incomplete: {path.name}')
            n = struct.unpack('<Q', raw)[0]
            if not 1 <= n <= 16 * 1024 * 1024 or path.stat().st_size < n + 8:
                raise FileNotFoundError(f'Download header incomplete: {path.name}')
            header = json.loads(f.read(n))
        expected = 8 + n + max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if path.stat().st_size != expected:
            raise FileNotFoundError(f'Download incomplete: {path.name}: {path.stat().st_size}/{expected} bytes')
    return paths


def code_identity():
    paths = list((ROOT / 'jev_train').glob('*.py'))
    paths += [ROOT / 'leaf_repro/common.py', ROOT / CONFIG['manifest']]
    paths += [ROOT / f'configs/jev13_{name}.json' for name in ('train', 'tasks', 'evaluation')]
    paths += list((ROOT / 'vendor/MedJev/medjev').glob('*.py'))
    paths += [Path(CONFIG['leaf_root']) / name for name in (
        'LEAF.py', 'load_config.py', 'configs/LEAF_mpnet.yaml', 'configs/tasks.yaml')]
    return {str(p): digest(p) for p in sorted(paths)}
