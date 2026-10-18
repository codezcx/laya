"""Deterministic, signal-derived questions; no class-label-based explanations."""
import numpy as np
from scipy.signal import welch

BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
FRONTAL = ['Fz', 'F3', 'F4', 'F7', 'F8']
POSTERIOR = ['Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2']
OPTIONS = {'A': 'The first is higher by more than 1 dB.',
           'B': 'The difference is within 1 dB (inclusive).',
           'C': 'The second is higher by more than 1 dB.'}


def relative_power(x):
    """Welch 1 Hz bins; normalize each channel/window by its 1–30 Hz power."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 200 or not np.isfinite(x).all():
        raise ValueError('Expected finite [batch, channel, time], at least 200 samples')
    f, p = welch(x, fs=200, window='hann', nperseg=200, noverlap=100,
                 nfft=200, detrend='constant', scaling='density', axis=-1)
    total = p[..., (f >= 1) & (f <= 30)].sum(-1)
    if np.any(total <= 0):
        raise ValueError('Zero-power signal')
    result = {}
    for band, (low, high) in BANDS.items():
        keep = (f >= low) & ((f <= high) if band == 'beta' else (f < high))
        result[band] = p[..., keep].sum(-1) / total
    return result


def specifications():
    specs = []
    def add(task, name, family, a, b, first, second):
        specs.append(dict(id=f'{task}_{name}', task=task, family=family,
                          first=a, second=b,
                          question=f'Compare {first} with {second}. '
                          'Use channel-wise relative band power normalized by 1–30 Hz power, '
                          'and average relative powers across channels for a region. '
                          'Which comparison holds?'))
    def side(channels, band, window='full'):
        return dict(channels=channels, band=band, window=window)
    for channel in ['C3', 'C4']:
        add('mi', f'spectral_{channel.lower()}', 'spectral',
            side([channel], 'alpha'), side([channel], 'beta'),
            f'{channel} 8–13 Hz (μ) relative power', f'{channel} 13–30 Hz (β) relative power')
    for band, label in [('alpha', '8–13 Hz (μ)'), ('beta', '13–30 Hz (β)')]:
        add('mi', f'spatial_{band}', 'spatial', side(['C3'], band), side(['C4'], band),
            f'C3 {label} relative power', f'C4 {label} relative power')
    for channel in ['C3', 'C4']:
        add('mi', f'temporal_{channel.lower()}_alpha', 'temporal',
            side([channel], 'alpha', 'late'), side([channel], 'alpha', 'early'),
            f'{channel} 8–13 Hz (μ) relative power during 1.5–3 s',
            f'{channel} 8–13 Hz (μ) relative power during 0–1.5 s')
    for name, channels in [('frontal', FRONTAL), ('posterior', POSTERIOR)]:
        region = '/'.join(channels)
        add('emotion', f'spectral_{name}', 'spectral', side(channels, 'alpha'), side(channels, 'beta'),
            f'region [{region}] 8–13 Hz (α) relative power',
            f'region [{region}] 13–30 Hz (β) relative power')
    for band, label in [('alpha', '8–13 Hz (α)'), ('beta', '13–30 Hz (β)')]:
        add('emotion', f'spatial_{band}', 'spatial', side(FRONTAL, band), side(POSTERIOR, band),
            f'region [{"/".join(FRONTAL)}] {label} relative power',
            f'region [{"/".join(POSTERIOR)}] {label} relative power')
        add('emotion', f'temporal_frontal_{band}', 'temporal',
            side(FRONTAL, band, 'late'), side(FRONTAL, band, 'early'),
            f'region [{"/".join(FRONTAL)}] {label} relative power during 1.5–3 s',
            f'the same region/band relative power during 0–1.5 s')
    return specs


def compare(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError('Power must be nonnegative')
    db = 10 * np.log10(np.maximum(first, 1e-12) / np.maximum(second, 1e-12))
    # Fixed descriptive tolerance, not a statistical significance criterion.
    answers = np.where(db > 1, 'A', np.where(db < -1, 'C', 'B'))
    return db, answers


def measure(powers, channels, spec):
    def value(side):
        indices = [channels.index(c) for c in side['channels']]
        return powers[side['window']][side['band']][:, indices].mean(1)
    a, b = value(spec['first']), value(spec['second'])
    db, answer = compare(a, b)
    return a, b, db, answer
