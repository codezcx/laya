"""Current MI normalization helper; historical experiment removed."""
import numpy as np
from .data import normalized_window

def scale_window(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (3, 600) or not np.isfinite(x).all():
        raise ValueError('Expected finite three-channel three-second window')
    if mode == 'per_channel':
        return normalized_window(x)
    if mode != 'common_scale':
        raise ValueError(mode)
    x = x - x.mean(-1, keepdims=True)
    if np.any(x.std(-1) < 1e-12):
        raise ValueError('Flat EEG channel')
    return np.clip(x / np.sqrt(np.mean(x * x)), -15, 15).astype(np.float32)
