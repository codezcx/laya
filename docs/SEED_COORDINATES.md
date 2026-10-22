# SEED coordinate audit

Audit command: `laya/.venv-eeg/bin/python scripts/audit_seed_coordinates.py`.
Results: `artifacts/seed_coordinate_audit/audit.json`, including input hashes. No training coordinates have been replaced by this audit.

SEED and SEED-VII supply numerically identical 62-channel EEGLAB polar templates. The local REVE bank contains 60 names; CB1 and CB2 are absent. The bank safetensors and positions.json agree exactly. For 148 shared names, bank coordinates agree with installed MNE 1.12.1 standard_1005 coordinates to 4.83e-9 metres maximum.

## Verified conversion

For the supplied angle theta (degrees) and topographic radius r, let t=theta*pi/180 and a=pi*r. The unit-sphere coordinates in MNE axis orientation are:

```
x = sin(a) * sin(t)  # positive right
 y = sin(a) * cos(t) # positive anterior
 z = cos(a)         # positive superior
```

The formula matches `mne.channels.read_custom_montage(..., head_size=1.)` within 1e-12 for all 62 channels in both datasets. This validates angular conversion, not physical head size. Radius in .locs is a normalized topographic arc distance, not metres.

References: [EEGLAB readlocs specification](https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/readlocs.m), [MNE custom montage reader](https://mne.tools/stable/generated/mne.channels.read_custom_montage.html), [REVE position bank](https://huggingface.co/brain-bzh/reve-positions).

## Registration to the actual REVE bank

Fit only geometry of the 60 known electrodes, with no EEG signals or labels. Compare a rotation/uniform-scale/translation fit against a full affine least-squares fit. For each known electrode, refit on the other 59 to assess interpolation error:

| Fit | In-sample RMS | Leave-one-out RMS | LOO median | LOO maximum |
|---|---:|---:|---:|---:|
| Similarity | 8.14 mm | 8.52 mm | 6.37 mm | 19.03 mm |
| Affine | 6.10 mm | 6.58 mm | 5.50 mm | 15.26 mm |

Affine estimates in the bank coordinate frame (metres):

| Electrode | x | y | z |
|---|---:|---:|---:|
| CB1 | -0.04418016 | -0.10409982 | 0.00457365 |
| CB2 | 0.04529740 | -0.10358043 | 0.00443091 |

These are template-based estimates, not measured CB locations. LOO performance on the other electrodes does not establish the true error at CB1/CB2; similarity/affine disagreement also indicates model dependence. No alias such as CB1=O1 or CB1=PO7 is justified.

Proposed all-channel input policy: preserve the 60 original bank entries and supply the two explicitly tagged affine estimates directly to REVE's positions tensor. Freeze this geometry before training; record provenance and check sensitivity against the similarity fit without choosing geometry on test labels. This removes the technical name-lookup obstacle while retaining the approximation as an experimental limitation.

## Official versus local implementation audit

Checked the upstream repository directly at `elouayas/reve_eeg` commit `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7`. Its preprocessing scripts and task configurations include FACED; no SEED/SEED-VII dataset loader or CB1/CB2 handling was found in this snapshot. The paper's Table 1 and Appendix C.1 report FACED emotion recognition (32 channels, nine classes), not SEED. This does not establish that SEED was absent from all pretraining data or unpublished experiments.

The locally extended repository at `BSR-VideoMAE/reve_eeg` does contain a SEED/SEED-VII harness: `src/eval_harness_fewshot.py:68` defines `POSITION_ALIASES = {"CB1": "PO9", "CB2": "PO10"}`, applied at line 174. This harness was introduced in local commit `49f3e9b` and is absent from the checked upstream snapshot. Its alias policy is a local approximation, not evidence of an official REVE coordinate-conversion method. The affine registration above is also our own alternative, not an author-supplied recipe.

Sources: [paper Appendix C.1](https://arxiv.org/html/2510.21585v1#A3.SS1), [pinned official preprocessing directory](https://github.com/elouayas/reve_eeg/tree/06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7/preprocessing).
