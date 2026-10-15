import ast
import io
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from tusz_sft.data import merge_intervals, read_annotations, windows_for_record
from tusz_sft.render import (CHANNEL_NAMES, common_average_uv, plot_eeg_data,
                             select_channels)


class AnnotationTests(unittest.TestCase):
    def test_event_union_and_exact_boundary(self):
        intervals = merge_intervals([[10, 12], [11, 19], [19, 20]])
        record = {"split": "train", "subject": "a", "edf": "a.edf",
                  "duration_seconds": 35, "seizures": intervals}
        windows = list(windows_for_record(record))
        self.assertEqual([w["label"] for w in windows], ["bckg", "seiz", "bckg"])
        self.assertEqual([w["seizure_seconds"] for w in windows], [0, 10, 0])
        self.assertFalse(any(w["boundary"] for w in windows))

    def test_sparse_annotation_and_bad_annotation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.csv_bi"
            p.write_text("# duration = 30.0000 secs\nchannel,start_time,stop_time,label,confidence\nTERM,1,2,seiz,1\n")
            duration, intervals = read_annotations(p)
            windows = list(windows_for_record({"split": "train", "subject": "a", "edf": "a.edf",
                                                "duration_seconds": duration, "seizures": intervals}))
            self.assertEqual([w["label"] for w in windows], ["seiz", "bckg", "bckg"])
            self.assertTrue(windows[0]["boundary"])
            p.write_text(p.read_text() + "TERM,29,31,seiz,1\n")
            with self.assertRaises(ValueError):
                read_annotations(p)


class SignalTests(unittest.TestCase):
    def test_reference_and_microvolt_scaling(self):
        volts = np.full((19, 2000), 5e-6)
        volts[0] += 19e-6
        uv = common_average_uv(volts)
        np.testing.assert_allclose(uv[0], 18, atol=1e-10)
        np.testing.assert_allclose(uv[1:], -1, atol=1e-10)
        np.testing.assert_allclose(uv.mean(axis=0), 0, atol=1e-10)

    def test_channel_order_missing_and_duplicate(self):
        names = ["EEG " + n.replace("-AV", "-REF") for n in CHANNEL_NAMES]
        self.assertEqual(select_channels(list(reversed(names)) + ["EEG EKG1-REF"]), names)
        with self.assertRaises(ValueError):
            select_channels(names[:-1])
        with self.assertRaises(ValueError):
            select_channels(names + ["EEG FP1-LE"])

    def test_pixel_parity_with_pinned_upstream(self):
        # Execute ONLY the audited plotting function, never the upstream file's
        # top-level filesystem scan or directory creation.
        upstream = Path(__file__).resolve().parents[1] / "third_party/CerebraGloss/CerebraGlossBench/scripts/build_img_benchmark.py"
        tree = ast.parse(upstream.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plot_eeg_data")
        scope = {"np": np, "Axes": Axes, "SFREQ": 200, "display_channel_width": 200}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(upstream), "exec"), scope)
        t = np.arange(2000) / 200
        data = np.array([30 * np.sin(2 * np.pi * (i + 1) * t) for i in range(19)])
        images = []
        for draw in (scope["plot_eeg_data"], plot_eeg_data):
            with plt.rc_context(plt.rcParamsDefault):
                fig, ax = plt.subplots(figsize=(6, 7.6))
                draw(ax, data, CHANNEL_NAMES)
                buf = io.BytesIO()
                fig.savefig(buf, format="jpg", dpi=200, bbox_inches="tight", pad_inches=0.1)
                images.append(buf.getvalue())
                plt.close(fig)
        self.assertEqual(images[0], images[1])


if __name__ == "__main__":
    unittest.main()
