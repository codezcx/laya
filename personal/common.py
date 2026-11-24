import json
from pathlib import Path

from jev_train.runtime import ROOT, code_identity, digest, write_json

CONFIG_PATH = ROOT / 'configs/personal13.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
OUT = ROOT / CONFIG['output_root']


def identity():
    result = code_identity()
    files = list((ROOT / 'personal').glob('*.py')) + [CONFIG_PATH]
    files += [Path('/inspire/hdd/global_user/zhangchenxi-253108310322/LEAF/leaf_datasets') / f'{name}.py'
              for name in ('MI_BCIC_IV2a', 'MI_PhysioNet', 'MI_OpenBMI')]
    return {**result, **{str(p): digest(p) for p in sorted(files)}}


def check_rows(rows, case, gradient=False):
    if CONFIG['original_split'] != 'valid' or not CONFIG['authorization']:
        raise RuntimeError('Personalization requires explicit valid-support authorization')
    allowed = set(case['support_ids'] if gradient else case['support_ids'] + case['query_ids'])
    if not rows or any(r['split'] != 'valid' or r['subject'] != case['subject']
                       or r['task'] != case['task'] or r['id'] not in allowed for r in rows):
        raise RuntimeError('Personal forward/gradient crossed its authorized support/query boundary')
    if set(case['support_ids']) & set(case['query_ids']):
        raise RuntimeError('Personal support/query overlap')
