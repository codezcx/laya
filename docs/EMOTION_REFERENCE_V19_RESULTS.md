# Amplitude-preserving common-reference control v19 results

All four extraction shards and eight source-selected fits completed. Two tests passed (shared-reference cancellation, common-unit invariance). All model hashes froze before held-dataset scoring. See [protocol](EMOTION_REFERENCE_V19_PROTOCOL.md).

| Source → held dataset | Features | As-recorded balanced % | CAR30 balanced % | CAR minus control, pp |
|---|---|---:|---:|---:|
| SEED → FACED | full_linear | 38.15 | 38.11 | -0.03 |
| SEED → FACED | spatial_linear | 37.75 | 37.82 | 0.07 |
| FACED-NEMAR → SEED | full_linear | 43.41 | 39.59 | -3.81 |
| FACED-NEMAR → SEED | spatial_linear | 44.17 | 42.39 | -1.78 |

## Findings

Uniform30-channel average referencing does not improve transfer consistently. SEED→FACED is nearly unchanged, while FACED→SEED falls by3.81pp for full_linear and1.78pp for spatial_linear. Retain the as-recorded baseline for now; do not silently switch all experiments to CAR. This does not identify either dataset’s original reference or prove referencing irrelevant to other encoders.

Local reference metadata remain unknown: every FACED sidecar says n/a; the inspected SEED preprocessed README does not specify reference/units. The experiment tests a preprocessing intervention, not a confirmed acquisition mismatch.

The independently re-extracted as-recorded features reproduce v16 closely: mean absolute log-fraction difference is approximately5.03e-8 for SEED and1.17e-7 for FACED (max4.77e-7 and5.05e-5 respectively). Thus common unit scaling and clipping differences in this spectral path do not explain its large transfer gap. Tiny feature perturbations can change selected classifier predictions, so the control table uses freshly fitted models rather than copying old scores.

## Next unresolved signal choices

The emotion pipeline remains restricted to1–30Hz and3-second windows, choices inherited from the MI setup. Source-only tests of a wider emotion frequency range and longer within-trial context are more informative next than another output-head change. Keep both transfer directions and an identical preprocessing control. No claim is made that wider bands will help before testing.

## Scope and artifacts

Known-task emotion diagnostic using repeatedly examined development subjects, not a new blind test or unified zero-shot success. No MI or 2b/SEED-VII evaluation. Source validation chooses C; all reference modes are reported without choosing them on target performance.

Artifacts: `artifacts/emotion_reference_v19/features/` records per-file sampling rate, units, reference metadata and manifest hashes. `data_complete.json` records12,120 rows; `reproducibility.json` records cache comparisons. Each mode/source/arm retains source sweeps and model hashes; `evaluation/` retains all predictions, ordinary/balanced scores, bootstrap intervals, shuffled controls and paired reference differences.
