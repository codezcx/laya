# Common-montage spectral emotion transfer v16 results

Eight source-selected models completed, with all hashes frozen before transfer evaluation. Two feature-invariance/channel-alias tests passed. Protocol: [v16](EMOTION_TRANSFER_V16_PROTOCOL.md).

| Train → held dataset | Representation / classifier | Source validation balanced % | Held subject balanced % (95% CI) | Shuffled % |
|---|---|---:|---:|---:|
| SEED → FACED | three_linear | 37.56 | 33.53 (32.37–34.71) | 33.19 |
| SEED → FACED | full_linear | 45.26 | 38.16 (36.00–40.52) | 34.28 |
| SEED → FACED | spatial_linear | 48.07 | 37.75 (35.56–40.03) | 32.75 |
| SEED → FACED | spatial_rbf | 48.52 | 36.41 (34.33–38.58) | 33.53 |
| FACED-NEMAR → SEED | three_linear | 36.25 | 33.72 (32.02–35.15) | 33.76 |
| FACED-NEMAR → SEED | full_linear | 41.77 | 43.41 (40.28–46.93) | 32.93 |
| FACED-NEMAR → SEED | spatial_linear | 42.26 | 44.17 (40.54–48.00) | 32.81 |
| FACED-NEMAR → SEED | spatial_rbf | 41.46 | 38.19 (35.83–40.72) | 33.31 |

Chance balanced accuracy is 33.33%. Confidence intervals resample subjects, not windows. FACED has 24 development subjects; SEED has 12. All target rows belong to previously examined development pools. These are known-task exploratory transfer scores, not fresh blind evaluations and not automatic task recognition.

## Findings

- Full common-montage frequency features clearly outperform the three-channel controls in both transfer directions in this experiment. FACED → SEED rises from 33.72% to 43.41%; spatial centering reaches 44.17%.
- SEED → FACED is weaker: 38.16% with all channels. This asymmetry argues against presenting one direction as universal emotion generalization.
- Nonlinear RBF classification does not improve transfer over these linear baselines. Spatial centering has inconsistent gains, so no universal benefit is established.
- These results support retaining broad scalp coverage and frequency structure in the emotion pathway. They do not prove why REVE readouts underperform: montage, pooling, objective, and representation differ between experiments.
- These are feature baselines, not the requested unified encoder/text model. MI was not modified or evaluated this round. Do not expand to stress/arithmetic on the basis of these scores.

## Next experiment rationale

Use common-channel log-band fractions as an explicit auxiliary signal or fused input to the signal representation, with full-channel coverage and the ordinary spectral baseline retained. Select hyperparameters on source subjects only, and report both transfer directions. Before claiming improvement, compare matched input coverage and identical splits; do not attribute gains to semantic text alignment without a no-text control.

## Artifacts

`artifacts/emotion_transfer_v16/data.json` records channels, legacy aliases, band definitions, source hashes and feature checksum. Each source/arm directory contains all source validation candidates, selected model and checksum; `evaluation/` contains frozen hashes, held predictions, balanced/ordinary accuracy, subject bootstrap intervals and fixed shuffled controls.

Warnings across all source fits: 0.
