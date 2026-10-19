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
