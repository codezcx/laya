"""Pinned 13-dataset recipe; training hyperparameters inherited from 5611b49."""
import json
from pathlib import Path

from leaf_repro.common import sha256 as digest, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/jev13_train.json').read_text())


def arm_settings(arm):
    return CONFIG['arm_settings'][arm]


def backbone_files():
    root = ROOT / 'laya/JEV-27B-VL'
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    paths = [root / name for name in sorted(set(index['weight_map'].values()))]
    paths += [root / name for name in ('config.json', 'model.safetensors.index.json',
        'tokenizer.json', 'tokenizer_config.json', 'adapter_vllm/adapter_config.json',
        'adapter_vllm/decision_head.json', 'adapter_vllm/adapter_model.safetensors')]
    return paths


def code_identity():
    paths = list((ROOT / 'jev_train').glob('*.py'))
    paths += [ROOT / 'leaf_repro/common.py', ROOT / CONFIG['manifest']]
    paths += [ROOT / f'configs/jev13_{name}.json' for name in ('train', 'tasks', 'evaluation')]
    paths += [Path(CONFIG['leaf_root']) / name for name in (
        'LEAF.py', 'load_config.py', 'configs/LEAF_mpnet.yaml', 'configs/tasks.yaml')]
    return {str(p): digest(p) for p in sorted(paths)}
