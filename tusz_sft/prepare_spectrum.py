"""Render matched per-electrode spectrograms; preserve waveform window IDs/splits."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from .data import read_jsonl, write_jsonl
from .decision_data import RECIPES, file_sha256
from .render import CHANNELS
from .render_v3 import load_recording

RECIPE = {"version": RECIPES["spectrum"], "source_signal": "cerebragloss-local-v3-strict19",
          "method": "scipy.signal.spectrogram", "fs": 200, "window": "hann",
          "nperseg": 400, "noverlap": 350, "nfft": 400, "detrend": "constant",
          "scaling": "density", "frequency_limits_hz": [.5, 70],
          "color_limits_db": [-30, 40], "units": "dB re 1 microvolt^2/Hz",
          "colormap": "viridis", "per_image_normalization": False,
          "figure_inches": [12, 8], "dpi": 110}


def power_spectrogram(data):
    if data.shape != (19, 2000) or not np.isfinite(data).all():
        raise ValueError("Expected a finite 19 x 2000 microvolt window")
    f, t, psd = spectrogram(data, fs=200, window="hann", nperseg=400,
                            noverlap=350, nfft=400, detrend="constant", scaling="density", axis=-1)
    keep = (f >= .5) & (f <= 70)
    return f[keep], t, 10 * np.log10(np.maximum(psd[:, keep], 1e-12))


def render_spectrum(data, path):
    f, t, db = power_spectrogram(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(plt.rcParamsDefault):
        fig, axes = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
        try:
            for i, ax in enumerate(axes.flat):
                if i == 19:
                    ax.axis("off")
                    continue
                im = ax.pcolormesh(t, f, db[i], shading="nearest", cmap="viridis", vmin=-30, vmax=40,
                                   rasterized=True)
                ax.set_title(CHANNELS[i] + "-AV", fontsize=9)
                # STFT centers cover 1-9 s. Blank margins show the unsampled edges honestly.
                ax.set_xlim(0, 10)
                ax.set_ylim(.5, 70)
                ax.set_xticks([0, 5, 10])
                ax.set_yticks([1, 20, 40, 60])
                ax.tick_params(labelsize=7)
            fig.supxlabel("Time (s)", fontsize=10)
            fig.supylabel("Frequency (Hz)", fontsize=10)
            fig.tight_layout(rect=(.02, .02, .91, 1))
            color_axis = fig.add_axes([.935, .15, .015, .7])
            fig.colorbar(im, cax=color_axis, label="Power density (dB re 1 µV²/Hz)")
            fig.savefig(path, dpi=110, bbox_inches="tight")
        finally:
            plt.close(fig)


def render_record(job):
    root, out, rows = job
    signal, _ = load_recording(Path(root) / rows[0]["edf"], rows[0]["duration_seconds"])
    output = []
    for row in rows:
        path = Path(out) / "images" / (row["id"] + ".png")
        epoch = signal[:, int(row["start_seconds"] * 200):int(row["end_seconds"] * 200)]
        render_spectrum(epoch, path)
        output.append({**row, "recipe": RECIPE["version"], "image": str(path.resolve()),
                       "image_sha256": file_sha256(path), "waveform_image": row["image"],
                       "waveform_image_sha256": row["image_sha256"]})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waveform-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path(
        "/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/TUSZ"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--manifest-names", nargs="+", default=["train.fit.jsonl", "train.validation.jsonl"])
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Use an empty output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    manifests = [a.waveform_dir / name for name in a.manifest_names]
    rows = [r for path in manifests for r in read_jsonl(path)]
    groups = defaultdict(list)
    for row in rows:
        if row["recipe"] != RECIPES["waveform"]:
            raise ValueError("Expected local v3 waveforms")
        groups[row["edf"]].append(row)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = []
        for group in pool.map(render_record, [(str(a.data_root), str(a.out), r) for r in groups.values()]):
            results.extend(group)
            print(f"rendered {len(results)}/{len(rows)} spectrograms", flush=True)
    by_id = {r["id"]: r for r in results}
    fusion_dir = a.out / "fusion"
    fusion_dir.mkdir()
    for source in manifests:
        matched = [by_id[r["id"]] for r in read_jsonl(source)]
        write_jsonl(a.out / source.name, matched)
        fusion = [{**r, "recipe": RECIPES["fusion"],
                   "images": [r["waveform_image"], r["image"]],
                   "image_sha256s": [r["waveform_image_sha256"], r["image_sha256"]]} for r in matched]
        write_jsonl(fusion_dir / source.name, fusion)
    (a.out / "recipe.json").write_text(json.dumps({**RECIPE, "windows": len(rows),
        "matched_source_sha256": {p.name: file_sha256(p) for p in manifests}}, indent=2) + "\n")
    print(f"Complete: {len(results)} matched windows", flush=True)


if __name__ == "__main__":
    main()
