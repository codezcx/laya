# Current MI experiment status

Current approach: full-channel, full-exported-duration left/right MI classification with frozen REVE and trainable JEV LoRA/connector/A-B head. Protocol: [PROTOCOL.md](PROTOCOL.md).

- Train:49,193 trials from9source datasets; validation:12,508 subject-disjoint trials.
- Test:6,520 trials from held-out2b. No test results yet.
- Outputs:`artifacts/mi_jev/`; live phase:`status.json`; train/validation:`run/metrics.jsonl`; best checkpoint:`run/best.json`; final test:`run/test_2b.json`.
- Full-channel token preparation and four-GPU smoke must pass before training. This file does not claim completed training or improved accuracy.

TUAB was stopped at user request at step265/725. Best source validation accuracy78.41% (step150). Its logs/checkpoints remain at`artifacts/tuab_jev`; stopped state and original implementation are archived in Git commit`7335cfd`. MI uses original JEV initialization, not TUAB trained weights.

Full-channel cache audit passed:68,221trials,49,193train/12,508validation/6,520heldout. Four-GPU two-step smoke completed with finite gradients and identical synchronized trainable-weight hashes across all ranks. Formal four-GPU run launched; initial full source validation precedes optimizer updates. No final accuracy is available yet.

## Matched frozen REVE linear probe

GPU4–7 completed32epochs with source-validation early stopping; selected epoch22. Only3,074linear parameters trained; fixed L/mid/R pooling and train-only standardization. Exact cache hash matches JEV.

| Split / metric | Value |
|---|---:|
| Overall validation accuracy | 68.16% |
| Mean per-dataset validation accuracy | 70.91% |
| Mean per-dataset validation BA | 70.91% |
| Held-out 2b accuracy | 50.15% |

| Source validation dataset | Trials | Accuracy |
|---|---:|---:|
| BNCI2014_001 | 576 | 83.68% |
| Chang2025 | 480 | 73.96% |
| Cho2017 | 2000 | 67.15% |
| Dreyer2023 | 4040 | 65.97% |
| Kumar2024 | 1600 | 61.44% |
| Lee2019_MI | 2200 | 70.50% |
| PhysionetMI | 993 | 69.99% |
| Weibo2014 | 300 | 64.00% |
| Zhou2016 | 319 | 81.50% |

Frozen REVE features support above-chance source held-subject classification under this fixed spatial readout. This does not establish successful cross-dataset transfer:2b accuracy is50.15%, with left recall100% and right recall0.31%. Do not use2b to tune this probe. Counts and preprocessing are matched to JEV, while readout/optimization differ.

JEV resumes onGPU0–3 from the latest saved optimizer checkpoint(step200); old log reached318, so201–318must be replayed. New output:`artifacts/mi_jev/resumed/`; original output is preserved.

## Whole-dataset 2a transfer control — completed

Frozen REVE, same fixed L/mid/R pooling and linear head recipe; fresh head/scaler. All2a removed from source train/validation. Sources:8datasets,47,177train and11,932validation. Final target:2,592trials from all9subjects of2a,22EEG channels.2b is not used. Train-only normalization; source-only model selection. Early stopping at40epochs selected epoch30.

| Metric | Accuracy |
|---|---:|
| Source validation overall | 68.30% |
| Source validation dataset mean | 69.60% |
| Held-out 2a overall / BA | 65.70% |

| 2a subject | Trials | Accuracy |
|---|---:|---:|
| 1 | 288 | 77.43% |
| 2 | 288 | 53.82% |
| 3 | 288 | 86.46% |
| 4 | 288 | 55.56% |
| 5 | 288 | 53.82% |
| 6 | 288 | 53.47% |
| 7 | 288 | 58.33% |
| 8 | 288 | 89.93% |
| 9 | 288 | 62.50% |

The frozen-feature linear readout shows partial cross-dataset transfer to2a, with substantial subject variation. This is not the earlier83.68%2a source-validation setting. Difference from2b cannot be attributed solely to channel count because target protocol/montage and source dataset membership also differ. This excludes supervised2a exposure for this probe, not possible REVE pretraining exposure.

Artifacts:`artifacts/mi_reve_probe_holdout_2a/{split_audit.json,manifest_train.jsonl,manifest_validation.jsonl,manifest_heldout.jsonl,metrics.jsonl,best.json,test_2a.json,complete.json}`. Ongoing JEV onGPU0–3 retains its original9-source protocol and is not a matched2a-heldout model.

## Paired2a input-channel ablation — completed

Same2,592trials, same frozen encoder, same epoch30held-out2a probe/scaler, same reference, duration and preprocessing. Both22-channel and3-channel signals were freshly encoded through REVE;3-channel features were not extracted from contextualized22-channel features. No fitting or threshold tuning.

| Input | Accuracy |
|---|---:|
| All22EEG channels | 65.70% |
| C3/Cz/C4 only | 51.31% |
| Difference (3−22) | -14.39percentage points |

| Subject | 22channels | 3channels |
|---|---:|---:|
| 1 | 77.43% | 50.35% |
| 2 | 53.82% | 50.00% |
| 3 | 86.46% | 50.00% |
| 4 | 55.56% | 50.35% |
| 5 | 53.82% | 50.00% |
| 6 | 53.47% | 50.00% |
| 7 | 58.33% | 50.00% |
| 8 | 89.93% | 50.00% |
| 9 | 62.50% | 61.11% |

Paired correctness:both correct769,only22correct934,only3correct561,both wrong328. Re-encoded22-channel decisions exactly matched all2,592original predictions; maximum probability drift0.000807 (the original cache was generated on the previous GPU architecture and stored asfp16). Preflight required <0.001probability drift and identical decisions; earlier overly strict numeric-check outputs are preserved under`artifacts/mi_reve_2a_channel_preflight/`.

Reducing this input montage toC3/Cz/C4 harms the current fixed readout and produces near-chance performance. This isolates the effect of the chosen channel reduction/spatial coverage for this model; it does not establish an inherent3-channel information limit or that channel count alone explains all2b differences. Final paired outputs:`artifacts/mi_reve_2a_channel_comparison/comparison.json` and`predictions_22ch.json`/`predictions_3ch.json`.


## Global mean REVE linear probe

Completed normally on GPU4–7, early stopping at epoch32; selected epoch22 by source macro dataset BA. Same nine-source train/validation manifests and frozen cache as the spatial probe; only Linear(512,2) trained after global valid-token averaging and training-only standardization.

| Readout | Source validation accuracy | Macro dataset accuracy | Held-out 2b accuracy |
|---|---:|---:|---:|
| Global mean, 512 features | 57.55% | 57.91% | 50.00% |
| Left/mid/right, 1536 features | 68.16% | 70.91% | 50.15% |

Global mean validation left/right recalls:54.41%/60.68%; validation loss0.6731. Dataset accuracies:2a58.16%, Chang54.79%, Cho64.15%, Dreyer55.12%, Kumar53.25%, Lee58.45%, Physionet58.91%, Weibo55.67%, Zhou62.70%. On all6,520held-out2b trials the global mean probe predicts left (left recall100%, right0%). These results show modest recoverable MI information without explicit spatial grouping on source held-out subjects, but no successful2b transfer. Spatial pooling improves overall source validation by10.62percentage points; grouping also changes feature dimension/readout capacity, so this is not an isolated causal test of spatial semantics.

Artifacts:`artifacts/mi_reve_probe_mean/{metrics.jsonl,best.json,validation_best.json,test_2b.json,complete.json}`. Training-loop time about12seconds excludes feature extraction/startup; frozen tokens were reused, not re-encoded.
