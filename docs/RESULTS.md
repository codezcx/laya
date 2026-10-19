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
