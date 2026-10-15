"""Local AAAI CerebraGloss v3 rendering and its EDF preprocessing.

Adapted from CerebraGloss/DataEngine/{tusz_adapter,render_images_v3}.py.
Unlike that adapter, this dataset requires all 19 electrodes (no zero imputation).
The signal operations and plot are identical for complete 19-channel records.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .render import CHANNELS, SFREQ, WINDOW_SECONDS, select_channels

RECIPE = {
    "version": "cerebragloss-local-v3-strict19",
    "channels": [ch + "-AV" for ch in CHANNELS],
    "reference": "mean of 19 electrodes, float32 microvolts before subtraction",
    "sfreq_hz": SFREQ, "window_seconds": WINDOW_SECONDS,
    "resampling": "MNE FFT, npad=auto, whole recording before epoching",
    "bandpass_hz": None, "notch_hz": None, "amplitude_units": "microvolts",
    "per_epoch_normalization": False, "negative_up": True,
    "channel_spacing_uv": 200, "figure_inches": [12, 8],
    "dpi": 110, "format": "png", "missing_channels": "reject recording",
    "label_policy": "any strictly positive seizure overlap in [start, end)",
}


def load_recording(edf_path, expected_duration=None):
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    selected = select_channels(raw.ch_names)
    units = {ch: raw._orig_units.get(ch) for ch in selected}
    if any(u not in {"µV", "uV", "mV", "V"} for u in units.values()):
        raise ValueError(f"Unexpected EDF units: {units}")
    source_sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / source_sfreq
    if expected_duration is not None and abs(duration - expected_duration) > max(.02, 1 / source_sfreq):
        raise ValueError("EDF duration disagrees with annotation")
    raw.pick(selected)
    if source_sfreq != SFREQ:
        raw.resample(SFREQ, method="fft", npad="auto", n_jobs=1, verbose="ERROR")
    data = raw.get_data(units="uV").astype(np.float32)
    data -= data.mean(axis=0, keepdims=True)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite signal")
    return data, {"source_sfreq_hz": source_sfreq, "source_channels": selected,
                  "source_units": units, "duration_seconds": duration}


def render_epoch(data_uv, output):
    if data_uv.shape != (19, 2000) or not np.isfinite(data_uv).all():
        raise ValueError("Expected finite (19, 2000) microvolt array")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(plt.rcParamsDefault):
        off = np.arange(19, 0, -1) * 2.6
        t = np.arange(2000) / SFREQ
        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            for i in range(19):
                s = data_uv[i].astype(float)
                if s.std() < 1e-6:
                    ax.plot(t, np.full_like(t, off[i]), "0.8", lw=.4)
                else:
                    ax.plot(t, -s / 100.0 * 1.3 + off[i], "k", lw=.6)
            ax.set_yticks(off)
            ax.set_yticklabels([ch + "-AV" for ch in CHANNELS], fontsize=8)
            ax.set_xlim(0, 10)
            ax.set_xlabel("Time (s)")
            ax.set_ylim(0, off[0] + 3)
            fig.tight_layout()
            fig.savefig(output, dpi=110, bbox_inches="tight")
        finally:
            plt.close(fig)
