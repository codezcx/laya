"""CerebraGloss-Bench rendering, with a separate, explicit EDF adapter.

Plotting adapted from iewug/CerebraGloss, commit
0d47617d5502a5260533d6aceb36eb29acf66377, MIT, Copyright (c) 2026 Gu Wei.
See ../third_party/CerebraGloss/LICENSE. EDF preprocessing is our addition,
not an assertion about the authors' unpublished preprocessing pipeline.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
            "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ")
CHANNEL_NAMES = tuple(f"{ch}-AV" for ch in CHANNELS)
SFREQ = 200
WINDOW_SECONDS = 10
CHANNEL_WIDTH_UV = 200
RECIPE = {
    "version": "cerebragloss-bench-tusz-v2",
    "upstream_commit": "0d47617d5502a5260533d6aceb36eb29acf66377",
    "channels": list(CHANNEL_NAMES), "reference": "mean of these 19 scalp electrodes",
    "sfreq_hz": SFREQ, "window_seconds": WINDOW_SECONDS,
    "resampling": "MNE polyphase, anti-aliasing; whole recording before epoching",
    "bandpass_hz": [0.5, 70.0], "filter": "MNE zero-phase FIR, firwin, whole recording",
    "notch_hz": None,
    "amplitude_units": "microvolts", "per_epoch_normalization": False,
    "negative_up": True, "channel_spacing_uv": CHANNEL_WIDTH_UV,
    "figure_inches": [6, 19 * 0.4], "dpi": 200, "format": "jpg",
    "label_policy": "any strictly positive seizure overlap in [start, end)",
}


def select_channels(names):
    """Require unique referential electrodes; never silently impute a channel."""
    found = {}
    aliases = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    for name in names:
        match = re.fullmatch(r"(?:EEG\s+)?([A-Z0-9]+)-(REF|LE|AR|AV)", name.strip().upper())
        if match:
            channel = aliases.get(match[1], match[1])
            if channel in CHANNELS:
                if channel in found:
                    raise ValueError(f"Duplicate electrode {channel}: {found[channel]}, {name}")
                found[channel] = name
    missing = set(CHANNELS) - set(found)
    if missing:
        raise ValueError(f"Missing referential scalp electrodes: {sorted(missing)}")
    refs = {name.rsplit("-", 1)[-1].upper() for name in found.values()}
    if len(refs) != 1:
        raise ValueError(f"Mixed reference suffixes: {refs}")
    return [found[ch] for ch in CHANNELS]


def read_edf_channel_names(path):
    """Read only signal labels, without loading EEG or exposing patient headers."""
    with Path(path).open("rb") as f:
        fixed = f.read(256)
        count = int(fixed[252:256])
        if not 1 <= count <= 1024:
            raise ValueError("Invalid EDF signal count")
        labels = f.read(count * 16)
    if len(labels) != count * 16:
        raise ValueError("Truncated EDF channel labels")
    return [labels[i * 16:(i + 1) * 16].decode("ascii").strip() for i in range(count)]


def common_average_uv(volts):
    data = np.asarray(volts, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != 19 or not np.isfinite(data).all():
        raise ValueError("Expected finite 19-channel data in volts")
    return (data - data.mean(axis=0, keepdims=True)) * 1e6


def load_recording(edf_path, expected_duration=None):
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    selected = select_channels(raw.ch_names)
    # Reject absent/unknown physical units rather than silently scale incorrectly.
    source_units = {name: raw._orig_units.get(name) for name in selected}
    if any(unit not in {"µV", "uV", "mV", "V"} for unit in source_units.values()):
        raise ValueError(f"Unexpected EDF voltage units: {source_units}")
    raw.pick(selected).load_data(verbose="ERROR")
    source_sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / source_sfreq
    if expected_duration is not None and abs(duration - expected_duration) > max(0.02, 1 / source_sfreq):
        raise ValueError(f"EDF duration {duration} != annotation duration {expected_duration}")
    if source_sfreq <= 140:
        raise ValueError(f"Sampling rate {source_sfreq} cannot support the fixed 70 Hz low-pass")
    # This is our explicit TUSZ preprocessing addition. Filtering the recording
    # before slicing avoids introducing a filter edge at every 10-second epoch.
    raw.filter(0.5, 70.0, picks=list(range(19)), method="fir", phase="zero",
               fir_design="firwin", n_jobs=1, verbose="ERROR")
    # MNE exposes physical amplitudes in volts, irrespective of EDF header units.
    if source_sfreq != SFREQ:
        raw.resample(SFREQ, method="polyphase", n_jobs=1, verbose="ERROR")
    data = common_average_uv(raw.get_data())
    return data, {"source_sfreq_hz": source_sfreq, "source_channels": selected,
                  "source_units": source_units, "duration_seconds": duration}


def plot_eeg_data(ax, data, channel_names=CHANNEL_NAMES):
    """Keep plot geometry and drawing parameters identical to upstream."""
    offset = np.arange(data.shape[0], 0, -1) * CHANNEL_WIDTH_UV
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1]) / SFREQ, -data[i] + offset[i],
                label=channel_names[i], color="black", linewidth=1)
    bar_y_start = offset[0] - CHANNEL_WIDTH_UV / 2
    bar_y_end = offset[0] + CHANNEL_WIDTH_UV / 2
    ax.plot([0, 0], [bar_y_start, bar_y_end], color="red", linewidth=1, zorder=10)
    ax.text(-0.1, bar_y_start, f"{CHANNEL_WIDTH_UV} µV", color="red",
            ha="right", va="center", fontsize=6)
    ax.set_xticks(np.arange(0, data.shape[1] / SFREQ + 1, 1))
    ax.set_xlim(0, 10)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_yticks(offset)
    ax.set_yticklabels(list(channel_names), fontsize=8)
    ax.set_ylim(CHANNEL_WIDTH_UV // 2, data.shape[0] * CHANNEL_WIDTH_UV + CHANNEL_WIDTH_UV // 2)
    ax.set_xlabel("Time (s)", fontsize=8)


def render_epoch(data_uv, output):
    if data_uv.shape != (19, SFREQ * WINDOW_SECONDS) or not np.isfinite(data_uv).all():
        raise ValueError(f"Expected finite (19, 2000) array, got {data_uv.shape}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(plt.rcParamsDefault):
        fig, ax = plt.subplots(figsize=(6, 19 * 0.4))
        try:
            plot_eeg_data(ax, data_uv)
            fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.1)
        finally:
            plt.close(fig)
