"""TUSZ term annotations, fixed epochs, and label-only instruction records."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

QUESTION = ("Does this 10-second EEG segment contain any seizure activity? "
            "Choose one option.\nA) Non-seizure\nB) Seizure\nAnswer only A or B.")
STATE = ("A 10-second EEG waveform with 19 scalp channels in common-average "
         "reference, sampled at 200 Hz. Channel spacing represents 200 microvolts; "
         "negative voltage is plotted upward.")


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def read_annotations(path):
    text = Path(path).read_text()
    match = re.search(r"^#\s*duration\s*=\s*([\d.]+)\s+secs", text, re.M)
    if not match:
        raise ValueError(f"Missing duration: {path}")
    duration = float(match[1])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration: {path}")
    lines = "\n".join(line for line in text.splitlines() if line.strip() and not line.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(lines)))
    if not rows:
        raise ValueError(f"Empty annotation file, not assumed background: {path}")
    seizures, background = [], []
    for row in rows:
        start, end = float(row["start_time"]), float(row["stop_time"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.001):
            raise ValueError(f"Invalid interval in {path}: {row}")
        if row["channel"] != "TERM" or row["label"] not in {"seiz", "bckg"}:
            raise ValueError(f"Unexpected term annotation in {path}: {row}")
        (seizures if row["label"] == "seiz" else background).append([start, min(end, duration)])
    seizures = merge_intervals(seizures)
    if any(min(s1, b1) > max(s0, b0) for s0, s1 in seizures for b0, b1 in background):
        raise ValueError(f"Conflicting seizure/background intervals: {path}")
    # TUSZ v2 stores sparse seizure events; unlisted time is non-seizure.
    # Background-only files may contain just a short bckg marker.
    return duration, seizures


def overlap_seconds(start, end, intervals):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


def windows_for_record(record):
    for index in range(int((record["duration_seconds"] + 1e-7) // 10)):
        start, end = index * 10, (index + 1) * 10
        overlap = overlap_seconds(start, end, record["seizures"])
        yield {"id": hashlib.sha256(f'{record["edf"]}:{start}'.encode()).hexdigest()[:24],
               "split": record["split"], "subject": record["subject"], "edf": record["edf"],
               "duration_seconds": record["duration_seconds"],
               "start_seconds": start, "end_seconds": end,
               "label": "seiz" if overlap > 1e-8 else "bckg",
               "seizure_seconds": round(overlap, 6),
               "boundary": 1e-8 < overlap < 10 - 1e-8}


def chat_record(window, image_path):
    """The prompt includes no patient ID, recording time, class or annotations."""
    return {"id": window["id"], "split": window["split"],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": STATE + "\n" + QUESTION}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "B" if window["label"] == "seiz" else "A"}]}]}


def jev_record(window, image_path):
    return {"id": window["id"], "split": window["split"],
            "kind": "choice", "state": STATE, "image": str(image_path),
            "question": "Does this 10-second EEG segment contain any seizure activity?",
            "options": ["Non-seizure", "Seizure"],
            "target_index": int(window["label"] == "seiz")}


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    with Path(path).open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
