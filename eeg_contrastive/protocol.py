"""Identity, label semantics and fixed text-bank construction; no dataset edits."""
import json
import math
from pathlib import Path

from eeg_align.runtime import CONFIG as DATA, digest
from eeg_align.benchmark_data import catalog
from eeg_align.online_bsr import upstream_files

CONFIG_PATH = Path('configs/contrastive_train.json')
CONFIG = json.loads(CONFIG_PATH.read_text())


def code_identity():
    paths = sorted(Path('eeg_align').glob('*.py')) + sorted(Path('eeg_contrastive').glob('*.py'))
    paths += [CONFIG_PATH, Path(CONFIG['data_config']), Path('configs/benchmark_tasks.json'),
              Path('scripts/run_contrastive.py'), Path('scripts/run_benchmark.py'), Path('scripts/prepare_multitask_signals.py'),
              Path('scripts/audit_multitask_signals.py')]
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
                candidate_policy=CONFIG['negatives'])


def rates_at(step, total, config=None):
    config = CONFIG if config is None else config
    warmup = min(config['warmup_steps'], max(1, total // 2))
    if step <= warmup:
        factor = step / warmup
    else:
        ratio = (step - warmup) / max(1, total - warmup)
        factor = config['min_lr_ratio'] + (1-config['min_lr_ratio']) * .5 * (1+math.cos(math.pi*ratio))
    return {name: value * factor for name, value in config['learning_rates'].items()}
