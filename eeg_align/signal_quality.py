"""Label-independent, immutable QC overlay; never alter signal caches or splits."""
import json
from pathlib import Path

import numpy as np

from .runtime import digest


POLICY = {
    'nonfinite': 'Any NaN or infinity in an epoch',
    'all_zero': 'Every channel and time point is exactly zero',
    'flatline': 'Every channel is temporally constant (including distinct channel offsets)',
    'action': 'Exclude invalid TRAIN epochs only; fail on any invalid held-out epoch',
    'missing_channel': 'An individual zero channel is allowed if another channel varies',
}


def invalid_epochs(x):
    """Exclusive reasons in priority order; no labels, thresholds, or fitted stats."""
    x = np.asarray(x)
    if x.ndim != 3 or min(x.shape[1:]) == 0:
        raise ValueError(f'Expected nonempty [epoch, channel, time], got {x.shape}')
    finite = np.isfinite(x).all(axis=(1, 2))
    zero = (x == 0).all(axis=(1, 2))
    flat = (x.max(axis=-1) == x.min(axis=-1)).all(axis=-1)
    reasons = np.full(len(x), '', dtype='<U10')
    reasons[flat] = 'flatline'
    reasons[zero] = 'all_zero'
    reasons[~finite] = 'nonfinite'
    return reasons


def excluded_training_indices(frame, exclusions):
    """Validate stable original CSV positions; held-out denominators cannot change."""
    excluded = set()
    for item in exclusions:
        index = item['index']
        if not isinstance(index, int) or not 0 <= index < len(frame) or index in excluded:
            raise ValueError('Invalid or duplicate QC exclusion index')
        original = frame.iloc[index]
        for key in ('array', 'row', 'subject', 'split', 'label', 'record_id'):
            if str(original[key]) != str(item[key]):
                raise ValueError(f'QC row identity changed: {index} {key}')
        if original['split'] != 'train':
            raise ValueError('QC must not silently exclude validation/test epochs')
        if item['reason'] not in {'nonfinite', 'all_zero', 'flatline'}:
            raise ValueError('Unknown exclusion rule')
        excluded.add(index)
    return excluded


def load_quality(config):
    root = Path(config['quality_root'])
    path = root / 'summary.json'
    if digest(path) != config['quality_summary_sha256']:
        raise ValueError('Signal QC summary changed or is not pinned')
    summary = json.loads(path.read_text())
    if not summary['training_ready'] or summary['policy'] != POLICY:
        raise ValueError('Signal QC incomplete, blocked, or policy changed')
    for filename, expected in summary['code_sha256'].items():
        if digest(filename) != expected:
            raise ValueError(f'Signal QC code changed: {filename}')
    if summary['split_summary_sha256'] != config['split_summary_sha256']:
        raise ValueError('Signal QC belongs to another subject split')
    if summary['cache_root'] != str(Path(config['cache_root']).resolve()):
        raise ValueError('Signal QC belongs to another cache root')
    reports = {}
    for name, item in summary['datasets'].items():
        path = root / f'{name}.json'
        if digest(path) != item['report_sha256']:
            raise ValueError(f'Signal QC report changed: {name}')
        report = json.loads(path.read_text())
        folder = Path(config['cache_root']) / name
        if digest(folder / 'index.csv') != report['index_sha256']:
            raise ValueError(f'Signal QC epoch index changed: {name}')
        for array, info in report['arrays'].items():
            stat = (folder / array).stat()
            if [stat.st_size, stat.st_mtime_ns] != info['stat']:
                raise ValueError(f'Signal array changed since QC: {name}/{array}')
        reports[name] = report
    return reports
