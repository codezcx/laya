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
    specs.extend(mi_specifications())
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


MI_CHANNELS = ['C3', 'Cz', 'C4']
CHANGE_OPTIONS = {'A': 'Task power decreases by more than 1 dB.',
                  'B': 'The change is within 1 dB (inclusive).',
                  'C': 'Task power increases by more than 1 dB.'}
SIDE_OPTIONS = {'A': 'C3 has a more negative change by more than 1 dB.',
                'B': 'The changes differ by at most 1 dB.',
                'C': 'C4 has a more negative change by more than 1 dB.'}
CLASS_OPTIONS = {'A': 'Left-hand motor imagery.', 'B': 'Right-hand motor imagery.'}
SEGMENT_TEXT = ('Two EEG segments from the same trial are provided: pre-cue fixation '
                'baseline [-2.5, -0.5] seconds and motor imagery [0.5, 2.5] seconds '
                'relative to cue onset. Both use a shared amplitude scale. ')


def mi_specifications():
    specs = []
    for band, label in [('alpha', '8–13 Hz (mu)'), ('beta', '13–30 Hz (beta)')]:
        for channel in ['C3', 'C4']:
            specs.append(dict(id=f'mi_change_{channel.lower()}_{band}', task='mi',
                family='baseline_change', channel=channel, band=band, options=CHANGE_OPTIONS,
                question=SEGMENT_TEXT + f'How does {channel} {label} absolute band power change '
                'from baseline to imagery? Use 10 log10(task power / baseline power).'))
        specs.append(dict(id=f'mi_lateralization_{band}', task='mi', family='lateralization',
            band=band, options=SIDE_OPTIONS, question=SEGMENT_TEXT +
            f'Compare the C3 and C4 {label} changes, each measured as '
            '10 log10(task power / its own baseline power). Which change is more negative? '
            'This comparison does not imply that either channel decreased.'))
    specs.append(dict(id='mi_class', task='mi', family='classification', options=CLASS_OPTIONS,
        question=SEGMENT_TEXT + 'Which hand is the participant instructed to imagine moving?'))
    return specs


def paired_normalize(baseline, task):
    pair = np.stack([baseline, task]).astype(np.float64)
    if pair.shape != (2, 3, 400) or not np.isfinite(pair).all():
        raise ValueError('Expected finite baseline/task [3,400]')
    pair -= pair.mean(-1, keepdims=True)
    scale = float(np.sqrt(np.mean(pair ** 2)))
    if scale <= 0:
        raise ValueError('Zero-power pair')
    return (pair / scale).astype(np.float32), scale


def absolute_power(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 400 or not np.isfinite(x).all():
        raise ValueError('Expected finite two-second segment')
    f, p = welch(x, fs=200, window='hann', nperseg=200, noverlap=100,
                 nfft=200, detrend='constant', scaling='density', axis=-1)
    return {band: p[..., (f >= lo) & ((f <= hi) if band == 'beta' else (f < hi))].sum(-1)
            for band, (lo, hi) in BANDS.items() if band in ('alpha', 'beta')}


def paired_answers(pair, label):
    powers = absolute_power(pair)
    changes = {}
    for band, power in powers.items():
        if np.any(power <= 0):
            raise ValueError('Zero band power')
        changes[band] = 10 * np.log10(power[1] / power[0])
    result = {}
    for spec in mi_specifications():
        if spec['family'] == 'classification':
            if label not in (0, 1):
                raise ValueError('Unexpected MI label')
            answer, evidence = 'AB'[label], {'label_source': 'original cue annotation', 'label': label}
        else:
            band = spec['band']
            delta = changes[band]
            value = float(delta[MI_CHANNELS.index(spec['channel'])]) if spec['family'] == 'baseline_change' else float(delta[0]-delta[2])
            answer = 'A' if value < -1 else 'C' if value > 1 else 'B'
            evidence = {'baseline_band_power': powers[band][0].tolist(),
                        'task_band_power': powers[band][1].tolist(),
                        'channel_changes_db': delta.tolist(), 'decision_value_db': value}
        result[spec['id']] = (answer, evidence)
    return result
