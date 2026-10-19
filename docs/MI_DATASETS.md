# Mounted left/right MI data adapter and audit

This is preparatory data work; the active TUAB training is unchanged. Configuration: `configs/mi_data.json`; loader: `eeg_align/mi_data.py`; full index builder: `scripts/prepare_mi_index.py`; remaining-candidate audit: `scripts/audit_mi_candidates.py`.

## Scope and loading

Mounted processed root: `/inspire/dataset/moabb-processed/v1/EEG-processed`.
Only exact `left_hand` / `right_hand` labels are mapped to 0 / 1. Select C3, Cz, C4 in that order, retain float32 microvolts at 200 Hz. No silent channel substitution, padding, amplitude correction, normalization, or additional filtering. Mounted exports have already been filtered/resampled. Exported epochs are task windows; an early segment is not a verified resting baseline.

The source allowlist contains 11 datasets: BNCI2014_001 (2a), Cho2017, Dreyer2023, Lee2019_MI, Weibo2014, Zhou2016, PhysionetMI, Brandl2020, Kumar2024, Chang2025, GuttmannFlury2025_MI. BNCI2014_004 (2b) stays held out and cannot be loaded as a source. Dataset/subject keys define reproducible, disjoint train/validation splits; adding a dataset does not reshuffle existing splits. These are per-dataset subject identities, not proof that no person appears across independently collected studies.

2a retains 2,592 left/right trials, 1,296 per class; all labels were checked against the original 18 MAT files (1=left, 2=right, 3=feet, 4=tongue). Physionet requires imagined hand runs R04/R08/R12. Chang and Guttmann require MI-specific source provenance. This is not an endorsement of all files tagged `imagery` by the exporter.

```python
from eeg_align.mi_data import MITrials
train = MITrials('datasets/mi_index/train.jsonl', split='train')
sample = train[0]  # eeg: [3, time] in uV, label: 0/1, metadata
# Optional crop_seconds=(0, 3): relative to exported epoch, not verified cue onset.
```

Index creation (refuses to overwrite an existing output directory):

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 laya/.venv-eeg/bin/python scripts/prepare_mi_index.py
```

`MITrials` requires a successful `complete.json`, matching configuration and manifest hashes. Invalid trials or conflicting exact duplicates block completion. Generated indexes are ignored by Git. Old failed audit evidence is preserved in `artifacts/mi_dataset_audit/initial_index`; original mounted data are never edited.

## Confirmed problems and exclusions

- **Dreyer2023:** subject 13 / session 0 / run 5R6online and subject 14 / session 0 / run 2R3online contain 40 pairs of identical trials across all 27 channels, including 20 pairs with opposite labels. Quarantine both entire runs (80 trials); do not arbitrarily trust either label. Evidence: `artifacts/mi_dataset_audit/dreyer_conflict.json`. Original raw-data root cause remains unestablished. Searches of two earlier MI manifests did not find these run paths; this finding does not explain earlier training failure by itself.
- **Dreyer2023A/B/C:** subsets of the full dataset, not three extra independent sources. Avoid duplicate inclusion.
- **Schirrmeister2017, GuttmannFlury2025_ME, Wairagkar2018, Wu2020:** motor execution; exclude from pure MI training despite hand labels.
- **Shin2017A:** lacks C3/C4. Nearby electrodes must not be silently renamed.
- **GrosseWentrup2009:** numeric electrode names need authenticated montage mapping before three-channel loading.
- **Kaya2018:** 1-second exported epochs; incompatible with a longer-window recipe without a separate justified protocol.
- **Yang2025 / Yang2025_3C:** very low recorded amplitudes; exclude pending raw BDF/physical-unit verification. Do not assume a correction factor. These are paradigms of one study; subject independence is also unresolved.
- **Stieger2021:** processed mount has only two subjects; durations span 0.08–6.04 s, with 316 epochs shorter than one second. Neural-bench raw MAT data cover more subjects; that is another representation of the same study, not another dataset.
- **Zhou2020:** 8,423 of 15,090 left/right epochs lack the requested named channels; require per-file channel audit and authenticated mapping before promotion.
- **Forenzo2023:** durations span 1.24–4 s (62 epochs shorter than 3 s); authenticate MI (versus OSA/MIOSA) and exported epoch boundaries against the continuous task protocol before promotion.
- **HefmiIch2025:** investigate cohort/provenance and large-amplitude outliers before promotion.
- **Kumar2024:** 44 of 7,200 epochs are shorter than five seconds (minimum 4.145 s). A five-second crop must not silently pad these.

**Liu2024:** subjects 4 and 5 share 40 pairs of exactly identical signals across all 29 channels (same labels). Exclude pending raw provenance verification; a subject split could otherwise leak samples. Evidence is in the remaining-candidate duplicate report.

Beetl2021_A/B (100/180 trials) are candidates, not automatically admitted by a numerically clean audit. Small candidate datasets need explicit subject-split and protocol decisions. All numerical checks apply to selected channels (or all exported channels when the requested channels are missing), not to artifact-free clinical EEG quality. Passing finite/nonflat checks does not establish correct events, units, or useful MI signal.

2b's three bipolar derivations are not equivalent to monopolar C3/Cz/C4 in other datasets. Preserve this domain difference; do not invent a common reference. Some processed 2b `raw_name` fields were inferred and appear incorrect; raw provenance verification remains needed before treating this as a newly authenticated final test set. Historical exposure also means 2b is not a completely untouched benchmark.

## Audit evidence and protocol references

Current complete source scan: `datasets/mi_index/audit.json`. Remaining directory scan: `artifacts/mi_dataset_audit/remaining_audit.json`; per-trial evidence: `remaining_records.jsonl` in the same directory. Counts are actual local files, not published full dataset sizes. 31 directories have both hand labels; removing three Dreyer subset directories and four execution datasets leaves **24 candidate directories**, not 24 verified independent MI datasets.

- [BCI Competition IV original protocols](https://www.bbci.de/competition/iv/)
- [PhysioNet executed versus imagined runs](https://www.physionet.org/content/eegmmidb/1.0.0/)
- [Dreyer dataset/subsets](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Dreyer2023.html)
- [Wairagkar executed finger tapping](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wairagkar2018.html)
- [Wu repackaging metadata](https://zenodo.org/records/18961128)
- [Schirrmeister executed movements](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html)
- [Kaya short MI epochs](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kaya2018.html)
- [Forenzo task conditions](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Forenzo2023.html)
- [Zhou2020 montage investigation](https://github.com/NeuroTechX/moabb/issues/1120)

## Full remaining-directory scan

Scanned 114,067 left/right files from the remaining 19 directories (including execution controls and duplicate subsets). No unreadable, nonfinite, or flat selected-channel arrays were found. This does not supersede semantic exclusions above.

| Directory | Trials | Subjects | Missing named target channels | Duration (s) |
|---|---:|---:|---:|---|
| Beetl2021_A | 100 | 2 | 0 | 4–4 |
| Beetl2021_B | 180 | 3 | 0 | 4–4 |
| Dreyer2023A | 14312 | 60 | 0 | 5–5 |
| Dreyer2023B | 5040 | 21 | 0 | 5–5 |
| Dreyer2023C | 1440 | 6 | 0 | 5–5 |
| Forenzo2023 | 3747 | 25 | 0 | 1.24–4 |
| GrosseWentrup2009 | 3000 | 10 | 3000 | 7–7 |
| GuttmannFlury2025_ME | 2519 | 31 | 0 | 4–4 |
| HefmiIch2025 | 2940 | 37 | 0 | 10–10 |
| Kaya2018 | 11096 | 7 | 0 | 1–1 |
| Liu2024 | 2000 | 50 | 0 | 4–4 |
| Schirrmeister2017 | 6742 | 14 | 0 | 4–4 |
| Shin2017A | 1740 | 29 | 1740 | 10–10 |
| Stieger2021 | 4713 | 2 | 0 | 0.08–6.04 |
| Wairagkar2018 | 1104 | 14 | 0 | 3–3 |
| Wu2020 | 1114 | 6 | 0 | 4–4 |
| Yang2025 | 30591 | 51 | 0 | 4–4 |
| Yang2025_3C | 6599 | 11 | 0 | 4–4 |
| Zhou2020 | 15090 | 18 | 8423 | 0.56–5 |

## Completed source index

After quarantining 80 Dreyer trials: 72,359 source trials (57,701 train; 14,658 validation), plus 6,520 held-out 2b trials. Full selected-channel scan found zero unexpected read/shape/unit/nonfinite errors and zero remaining exact duplicates. All 2,592 retained 2a labels match original MAT labels. Seven loader tests passed, and actual train/validation/heldout samples loaded successfully with manifest/configuration hash verification. These checks establish adapter integrity, not event-level physiological validity for every dataset.
