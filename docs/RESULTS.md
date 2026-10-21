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

## MI JEV runtime diagnostic: initial findings

Both learning-rate arms use identical 72-trial manifests and identical initialization outputs (train fit accuracy50%, loss0.7350264). On the selected shortest84-token and longest310-token trials, individual versus mixed-length batch inference differs by at most0.00074923 in probability (0.074923percentage points); maximum raw-logit drift0.01958. Poisoning invalid input padding produces exactly zero logit difference. These limited initial checks do not show a material padding failure; they do not establish full training-gradient equivalence or rule out later checkpoint-specific issues. Per-arm `padding.json`, `fit_0000.json`, and forthcoming fitting curves are in `artifacts/mi_jev_diagnostic_{mi,tuab}/`. The60-step comparison has completed; results follow below. Original live JEV dependency hashes remain unchanged.


## Completed matched learning-rate diagnostic

Both arms completed60steps on the identical72training trials (batch16; seed42). Full-set fitting accuracy at steps0/10/20/30/40/50/60: MI recipe50/50/50/50/50/52.78/51.39%; TUAB recipe50/52.78/59.72/55.56/59.72/56.94/65.28%. Final full-set CE:0.68840 versus0.56783. Mean predicted right probability for true left/right: MI49.51%/50.07%; TUAB35.51%/58.45%. All connector/LoRA/head groups received nonzero finite gradients at the first and last logged steps.

This supports a contribution from the learning-rate/schedule recipe to early fitting failure, not a universal inability to read MI features. It does not isolate learning rate from warmup: the MI arm is still at60/115warmup steps, while TUAB uses full constant rates from step1. Neither arm demonstrates successful overfitting of the72samples;65.28% is training accuracy, not validation or cross-dataset transfer. Runs have one seed and no matched tiny-set linear baseline. Thus higher rates improve fitting in this diagnostic but do not establish the sole cause or a generalization fix. Final full-set outputs:`artifacts/mi_jev_diagnostic_mi/fit_0060.json` and`artifacts/mi_jev_diagnostic_tuab/fit_0060.json`. Original full MI run separately reports source-validation accuracy50.27% at400 and50.82% at600; these are distinct from fitting scores.

## External Emotiv zero-shot: source checkpoint2307

Completed all880trials with no target training/calibration. Overall accuracy34.09% (300/880), left recall17.50%, right50.68%; confusion [[77,363],[217,223]], AUROC0.2470. By original collection split/date: September17 (400trials)35.50%; September18 (400)34.25%; September20 (80)26.25%. All three splits were evaluation-only. This is unsuccessful transfer to one external subject; below-chance direction warrants future label/acquisition provenance checking but labels or thresholds were not flipped after seeing results. Original labels left=0/right=1 match source mapping.

Input14electrodes AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4; none of C3/Cz/C4. 636samples at128Hz resampled to994samples at200Hz, edge-padded to1100 yielding84REVE tokens. Clipped sample fraction0.00010665. Cortex uV interpretation is documented but acquisition export conversion remains independently unverified. No added filter/reference/EA. Artifacts:`artifacts/emotiv_jev_zero_shot/{config.json,metrics.json,predictions.json,complete.json}`. Do not attribute failure solely to missing central channels: device, reference, processing, cue/label semantics and subject differences are also unisolated.

## BEETL / Shin external zero-shot — completed

Fixed source-selected JEV step2307 (`mi_jev_tuab_lr_continued/best.pt`), all channels/full epochs, unchanged A=left/B=right prompt and threshold0.5; no target adaptation. Four GPUs evaluated all2,020trials in285.25seconds (evaluation-script timer including audit/model loading). All four independently generated input manifests matched. Full-channel finite/nonflat/coordinate checks passed. BEETL C3/Cz/C4 hashes did not overlap source train/validation; this hash comparison was not applicable to Shin's different montage. Absence from REVE pretraining is not established.

| Dataset | Trials / subjects | Channels | Accuracy / BA | Left recall | Right recall | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Beetl2021_A | 100 / 2 | 63 | 51.00% | 2.00% | 100.00% | 0.3992 |
| Beetl2021_B | 180 / 3 | 32 | 48.33% | 1.11% | 95.56% | 0.4467 |
| Shin2017A | 1740 / 29 | 30 | 51.26% | 15.29% | 87.24% | 0.5158 |

All datasets are class-balanced. Confusion matrices (rows true left/right, columns predicted left/right): A `[[1,49],[0,50]]`; B `[[1,89],[4,86]]`; Shin `[[133,737],[111,759]]`. Predicted-right fractions are99.00%,97.22%,85.98%. BEETL-A subject accuracies52%,50%; BEETL-B50%,50%,45%; Shin range36.67–73.33% (60trials/subject). Aggregate performance is near chance with strong right-answer bias. These results do not support reliable cross-dataset transfer, and AUROC near/below0.5 means a threshold issue alone is not established. Do not flip labels or tune thresholds on these targets.

Metrics including every subject: `artifacts/external_mi_jev_zero_shot/metrics.json`; individual predictions: `predictions.json`; exact input provenance: `manifest_rank0.jsonl`; checkpoint/preprocessing hashes: `config.json`; completion: `complete.json`. Runner: `scripts/evaluate_external_mi.py`; log: `artifacts/external_mi_jev_zero_shot.log`.

## Unfrozen REVE control — running

Started four-GPU joint encoder/head training with `eeg_align.train_reve`, implementation commit `d1cb94e`, on the identical nine-source JEV manifests. Mean token pooling plus Linear(512,2); all active REVE parameters trainable. The distributed smoke test passed: every active parameter received a finite synchronized gradient. Main run passed step90 normally; no validation result was available at this status update. Batch training accuracy is not fixed-model full-training accuracy. See `artifacts/mi_reve_finetune/metrics.jsonl`, `best.json` after first validation, and `artifacts/mi_reve_finetune.log`. Final results are pending; external-target results from frozen REVE must not be attributed to this run.
