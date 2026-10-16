"""Shared signal preprocessing retained by the current token cache."""
import numpy as np
from scipy.signal import butter,sosfiltfilt,resample_poly

def preprocess(x,fs,cfg):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite raw signal')
    if fs!=200:
        from math import gcd
        g=gcd(int(fs),200);x=resample_poly(x,200//g,int(fs)//g,axis=-1)
    x=sosfiltfilt(butter(4,cfg['bandpass'],btype='bandpass',fs=200,output='sos'),x,axis=-1)
    return x


def normalized_window(x):
    x=x-x.mean(-1,keepdims=True)
    if (x.std(-1)<1e-12).any(): raise ValueError('Flat EEG channel')
    # Same per-channel normalization across all data sources.
    return np.clip(x/x.std(-1,keepdims=True),-15,15).astype('float32')
