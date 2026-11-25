import json
from pathlib import Path
from jev_train.runtime import ROOT, code_identity, digest, write_json

CONFIG_PATH = ROOT / 'configs/personal13.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
OUT = ROOT / CONFIG['output_root']
PARENT = ROOT / CONFIG['parent']
PARENT_CONFIG = json.loads((PARENT / 'config.json').read_text())['config']


def identity():
    files = list((ROOT / 'personal').glob('*.py')) + [CONFIG_PATH, ROOT / CONFIG['catalog']]
    files += [PARENT / n for n in ('config.json', 'complete.json')]
    for dataset in CONFIG['datasets']:
        files += [ROOT / CONFIG['split_root'] / dataset / n for n in ('manifest.json', 'records.csv')]
        files += [ROOT / CONFIG['cache_root'] / dataset / n for n in ('index.csv', 'complete.json')]
    for key in ('paired_reference', 'matched_reference'):
        files += [ROOT / CONFIG[key] / name for name in ('plan.json', 'summary.json')]
    files += [Path(PARENT_CONFIG['leaf_root']) / n for n in (
        'leaf_datasets/__init__.py', 'leaf_datasets/electrode_unifier.py',
        'configs/LEAF-ch65/LEAF-ch65.json', 'configs/LEAF-ch65/LEAF-ch65.bvef')]
    return {**code_identity(), **{str(p): digest(p) for p in sorted(files)}}


def check_rows(rows, case, gradient=False):
    if not CONFIG['authorization'] or case['subject'] not in CONFIG['cohorts'][case['dataset']]['subjects']:
        raise RuntimeError('Personalization requires an authorized subject/dataset')
    if set(case['support_ids']) & set(case['query_ids']):
        raise RuntimeError('Personal support/query overlap')
    allowed = set(case['support_ids'] if gradient else case['support_ids'] + case['query_ids'])
    if not rows or any(r['subject'] != case['subject'] or r['task'] != case['task']
                      or r['id'] not in allowed or r['split'] != case['row_splits'][r['id']] for r in rows):
        raise RuntimeError('Personal forward/gradient crossed its support/query boundary')
