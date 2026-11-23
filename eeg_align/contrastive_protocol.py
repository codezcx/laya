"""Identity, label semantics and fixed text-bank construction; no dataset edits."""
import json
import math
from pathlib import Path
import numpy as np
import torch

from eeg_align.runtime import CONFIG as DATA, digest
from eeg_align.benchmark_data import catalog
from eeg_align.online_leaf import upstream_files

CONFIG_PATH = Path('configs/contrastive_train.json')
CONFIG = json.loads(CONFIG_PATH.read_text())


def code_identity():
    paths = sorted(Path('eeg_align').glob('*.py'))
    paths += [CONFIG_PATH, Path(CONFIG['data_config']), Path('configs/benchmark_tasks.json'),
              Path('scripts/run_contrastive.py'), Path('scripts/run_benchmark.py'), Path('scripts/prepare_multitask_signals.py'),
              Path('scripts/audit_multitask_signals.py'), Path('scripts/check_multitask_signals.py')]
    paths += upstream_files(DATA)
    return {str(p): digest(p) for p in paths}


def build_text_manifest(tasks=None, config=None):
    tasks = catalog() if tasks is None else tasks
    config = CONFIG if config is None else config
    texts, concepts, banks, lookup = [], [], {}, {}
    for task in tasks:
        indices, labels, ids = [], [], []
        for c, option in enumerate(task['classes']):
            if str(c) != option['class_id']:
                raise ValueError('Catalog label IDs must match cached contiguous labels')
            label = config['canonical_labels'].get(option['option'], option['option'])
            key = (task['instruction'], label)
            if key not in lookup:
                concept = len(concepts)
                concepts.append(dict(semantic_id=concept, instruction=key[0], label=key[1]))
                caption_ids = []
                for template in config['text_templates']:
                    caption_ids.append(len(texts))
                    texts.append(template.format(instruction=key[0].rstrip('.'), label=key[1]))
                lookup[key] = (concept, caption_ids)
            concept, caption_ids = lookup[key]
            indices.extend(caption_ids)
            labels.extend([c] * len(caption_ids))
            ids.append(concept)
        if len(ids) != len(set(ids)):
            raise ValueError('Two candidate classes map to the same concept')
        banks[task['task_id']] = dict(text_indices=indices, class_labels=labels, semantic_ids=ids)
    if len(set(texts)) != len(texts):
        raise ValueError('Duplicate templates in global bank')
    return dict(texts=texts, concepts=concepts, banks=banks,
                provenance='Class-label supervision; template variations are not new trial annotations',
                candidate_policy=config['negatives'])


def rates_at(step, total, config=None):
    config = CONFIG if config is None else config
    warmup = min(config['warmup_steps'], max(1, total // 2))
    if step <= warmup:
        factor = step / warmup
    else:
        ratio = (step - warmup) / max(1, total - warmup)
        factor = config['min_lr_ratio'] + (1-config['min_lr_ratio']) * .5 * (1+math.cos(math.pi*ratio))
    return {name: value * factor for name, value in config['learning_rates'].items()}


def load_text_bank(root=None):
    root = Path(CONFIG['text_bank_root'] if root is None else root)
    if digest(root / 'text_bank.json') != CONFIG['text_bank_sha256']:
        raise ValueError('Pinned text-bank manifest changed')
    bank = json.loads((root / 'text_bank.json').read_text())
    expected = build_text_manifest()
    if any(bank[k] != expected[k] for k in expected):
        raise ValueError('Semantic text bank differs from the task catalog')
    if bank['model_sha256'] != CONFIG['text_model_sha256']:
        raise ValueError('Text encoder provenance changed')
    if digest(root / 'text_vectors.npy') != bank['vectors_sha256']:
        raise ValueError('Text embedding cache changed')
    vectors = np.load(root / 'text_vectors.npy', allow_pickle=False)
    if (vectors.shape != (len(bank['texts']), CONFIG['projection_width'])
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError('Invalid normalized text bank')
    return bank, vectors


def parent_directory(objective):
    if objective not in CONFIG['arms']:
        raise ValueError('Unknown alignment objective')
    return Path('artifacts/benchmark') / CONFIG['run_name'] / 'alignment' / 'train' / objective


def load_aligned_tower(encoder, directory, objective, cache_indexes):
    """Transfer only a complete, same-split parent's Tower; never its projection."""
    from .checkpoint import tensor_fingerprint
    directory = Path(directory)
    complete = json.loads((directory / 'complete.json').read_text())
    identity = json.loads((directory / 'config.json').read_text())
    if (complete['smoke'] or complete['test_evaluations'] or complete['jev_loaded']
            or not complete['encoder_updated'] or not complete['checkpoint_reload_verified']
            or not complete['frozen_text_verified'] or complete['updates'] != CONFIG['steps']):
        raise ValueError('Parent is not a completed train-only alignment run')
    if (identity['arm'] != objective or identity['smoke'] or identity['config'] != CONFIG
            or identity['data_config'] != DATA or identity['cache_indexes'] != cache_indexes
            or identity['bank_sha256'] != CONFIG['text_bank_sha256']):
        raise ValueError('Alignment parent split/config/cache identity mismatch')
    saved = torch.load(directory / 'best.pt', map_location='cpu', mmap=True, weights_only=False)
    if saved['identity'] != identity or saved['step'] != complete['selected_step']:
        raise ValueError('Parent selected checkpoint identity mismatch')
    source = {}
    for group in ('trainable', 'buffers'):
        source.update({k.removeprefix('encoder.'): v for k,v in saved[group].items() if k.startswith('encoder.')})
    actual = encoder.state_dict()
    if (source.keys() != actual.keys()
            or any(source[k].shape != actual[k].shape or source[k].dtype != actual[k].dtype for k in actual)):
        raise ValueError('Parent Tower parameter/buffer layout mismatch')
    proof = tensor_fingerprint(source)
    if proof != complete['selected_encoder_sha256']:
        raise ValueError('Parent Tower fingerprint mismatch')
    encoder.load_state_dict(source, strict=True)
    return dict(objective=objective, parent=str(directory), selected_step=complete['selected_step'],
                selected_encoder_sha256=proof, parent_checkpoint_sha256=digest(directory / 'best.pt'),
                transferred='encoder parameters and buffers only; no projection/scale/bias/JEV/optimizer',
                parent_test_evaluations=0, cache_identity_verified=True)
