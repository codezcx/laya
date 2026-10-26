# Joint cross-subject LEAF/JEV results

## Current four-arm task-mixture diagnostic

Run root: `artifacts/benchmark/cross_subject_fold0_task_mixture/`. The current experiment is 2a only versus 2a + SEED, crossed with frozen versus fine-tuned LEAF. All arms start from release weights, use identical raw-preprocessed 2a inputs and eight identical 2a examples per update, fixed BN/dropout and unit task-loss coefficients. Budget: 600 updates, seed 42, fold 0. See [the current protocol](PROTOCOL.md).

Primary endpoint is final-step 2a validation BA; best-by-2a checkpoints are secondary. Initial and 100-step validation also records SEED, class recalls and option-reversal consistency. Single-task SEED scores are untrained-task diagnostics. No test evaluation, PCGrad or larger task sweep is queued.

Preparation: 35 unit tests passed, including complete 600-step anchor matching, non-diluted task-loss gradients and reproducible RNG fingerprints. A mixed-input check confirmed 2a shape 65×800 and SEED shape 65×200, with unchanged SEED input values. Real-model smoke and formal training status are recorded in the run root's `pipeline.json`; this document does not infer completion from process launch. New accuracy results are pending. `monitor.json`, `LIVE.md` and TensorBoard update during execution. The earlier protocol/results snapshot is preserved at Git `5200248`; ignored historical artifacts are unchanged.

## Completed preprocessing continuation (2026-10-26 audit)

Both `cross_subject_fold0_leaf_preprocessing` branches completed step 800. The table reports 2a validation accuracy on the same 288 trials, not test accuracy. Neither job remains active. This snapshot closes the prior experiment before the matched single-task/joint-task factorial control.

| Update | Original adapter | Raw LEAF preprocessing |
| --- | ---: | ---: |
| 600, before continuation | 27.43% | 25.35% |
| 700 | 26.39% | 39.24% |
| 800 | 24.65% | 32.99% |

Final eleven-dataset macro BA was 53.40% / 54.70%; selected macro BA was 55.59% at step 600 / 56.17% at step 700. Preprocessing plus adaptation helps 2a in this diagnostic but does not resolve multi-task interference. Evidence: each branch's `complete.json` and `validation_{0600,0700,0800}.json`. Earlier startup notes below are historical.

Historical paired continuation: `artifacts/benchmark/cross_subject_fold0_leaf_preprocessing/`. Both branches restored the same frozen-BN best checkpoint at step 600 and completed 200 updates to step 800; only `leaf_raw` replaced 2a preprocessing. The other ten tasks, weights/optimizer/RNG initialization and schedule matched. Its original protocol is archived at Git `5200248`; the current protocol above describes a different, explicitly matched task-mixture experiment.

Historical startup verification: 28 focused tests and both real-model smoke tests passed, including exact 38-class checkpoint reloads. A mixed batch verified byte-identical inputs for all ten unchanged tasks. Evidence: `startup_checks.json` and smoke logs under the preprocessing run root. Smoke scores are functional checks, not experiment results.

Both preprocessing initial snapshots matched the step-600 source: trainable parameters, buffers, optimizer, scheduler and RNG states (4,767 tensor pairs per branch). Evidence: `initial_comparison.json` and `reload_check_0600.json` in that historical run root.

## Raw LEAF preprocessing control on 2a

The same frozen initial encoder, 2,016 training trials and 288 validation trials were evaluated with the original online adapter versus the raw LEAF 2a pipeline. Encoder parameters, buffers, instruction text, IDs and labels match exactly. The native pipeline uses T-session MAT files only; selecting labeled runs preserves 96 A04T trials that upstream's fixed skip would drop. This is a bundled input-pipeline intervention, including subject-level unlabeled normalization; it is not an isolated CAR effect or a JEV training result.

| 2a metric | Current pipeline | Raw LEAF pipeline |
| --- | ---: | ---: |
| Fixed C=1 probe validation accuracy | 30.21% | 32.29% |
| Source-subject-CV-selected probe validation accuracy | 29.51% | 36.11% |
| Selected source-subject mean BA | 38.00% | 54.37% |
| Selected C | 0.01 | 0.0001 |
| Released projection/prototype validation accuracy | 25.69% | 26.74% |

Source-subject BA improves at every identical C in the fixed grid: at C=0.01 it changes from 38.00% to 50.55%, and at C=0.0001 from 34.47% to 54.37%. Thus the source-fold improvement is not solely a result of selecting a different C. The selected-probe outer validation improvement is 19/288 additional correct predictions, but the held-out subject is still weak and has been inspected repeatedly. No test data were evaluated. All probe fits converged and reloaded metrics match. Upstream's default FIR length (2,751 samples) exceeds its six-second epochs (1,500 samples); this behavior and its MNE warning were preserved for this comparison rather than silently changing the filter.

Evidence: `artifacts/benchmark/cross_subject_fold0_bn_control/preprocessing_control_2a/comparison.json`, prepared cache `datasets/benchmark/leaf_2a_preprocessing/`, and probe snapshots/predictions under `preprocessing_control_2a/leaf_raw_probe/`. TensorBoard includes both fixed-C and source-CV probe results as final split metrics, not iterative training curves.

## TensorBoard records

The completed BN comparison has been exported to `artifacts/benchmark/cross_subject_fold0_bn_control/tensorboard/`, with runs `updating`, `frozen` and `probes/*`. Each training branch has 1,100 training points and six validation points. Accuracy and loss are recorded overall and for all 11 datasets; BA is a separate metric. Validation macro metrics weight datasets equally, while `*_pooled` weights examples. Validation loss is mean negative log probability of the true semantic class, reconstructed from saved predictions. Accuracies are fractions in [0,1]. `scripts/log_tensorboard.py --run <root> --out <root>/tensorboard --follow` also supports live export, and the training launcher starts it automatically. JSON metrics remain the authoritative source.

## Completed BatchNorm comparison

Historical experiment: `artifacts/benchmark/cross_subject_fold0_bn_control/`. Two single-H200 branches compared updating vs fixed BN statistics from the same update-100 checkpoint; both unfroze LEAF and kept JEV LoRA active. Both completed the **1200-update diagnostic stage**, with unchanged 3000-step LR schedule. Best validation macro BA: **51.85% updating (step 1200) / 55.59% frozen (step 600)**. Final step-1200 macro BA: **51.85% / 55.39%**. All checkpoint reload checks passed; neither branch is running. These are validation results, not independent test scores.

- Live summaries: `LIVE.md`, `monitor.json` under the current root.
- Per-step metrics: `{updating,frozen}/metrics.jsonl`.
- Fixed monitor validation: 9,748 rows, every 200 global updates.
- Last checkpoint: every 100 updates; best: validation-selected dataset-macro BA.
- Historical artifact index: [ARCHIVE.md](ARCHIVE.md).

## Completed matched comparison

Both branches completed all 1,100 continuation updates (101–1200) and six scheduled validations on the same 9,748 IDs/labels. Saved probabilities reproduce the reported normal/reversed metrics; every validation reload check covers all 38 task/classes with exactly matching logits. The controller reports `complete`, both branch completion files record step 1200, and final/best checkpoint metadata are audited in `completion_audit.json`. Training losses and gradient norms are finite throughout, and each update contains eight examples from each dataset. No test subjects were evaluated or used for selection.

| Validation update | Updating BN macro BA | Frozen BN macro BA |
| --- | ---: | ---: |
| 200 | 46.25% | 54.53% |
| 400 | 47.40% | 55.38% |
| 600 | 51.48% | **55.59%** |
| 800 | 44.73% | 53.40% |
| 1000 | 44.63% | 54.22% |
| 1200 | **51.85%** | 55.39% |

Freezing BN improves macro BA at all six matched checkpoints in this run. Its observed validation range is 2.19 points, versus 7.22 with updating BN; the final paired difference is 3.54 points. This supports keeping frozen statistics as the better tested normalization policy for this setup. It is a one-seed, one-fold result and does not establish universal superiority. Training continues to fit better with frozen BN (last 100 updates: loss 0.84683, sampled accuracy 58.87%, versus 0.89572 / 56.08%), but validation has not surpassed its step-600 best.

| Dataset | Updating step 1200 accuracy / BA | Frozen step 1200 accuracy / BA | Frozen selected step 600 accuracy |
| --- | ---: | ---: | ---: |
| BNCI2014001 (2a) | 28.13% / 28.13% | 26.74% / 26.74% | 27.43% |
| BNCI2014004 (2b) | 52.50% / 52.50% | 50.00% / 50.00% | 50.83% |
| BNCI2015001 | 64.50% / 64.50% | 66.00% / 66.00% | 75.00% |
| BNCI2014008 (P300) | 50.00% / 52.42% | 55.68% / 52.64% | 44.49% |
| BNCI2014009 (P300) | 38.89% / 55.00% | 54.86% / 54.58% | 61.98% |
| CHB-MIT | 58.25% / 70.70% | 81.67% / 83.50% | 83.50% |
| TUAB | 56.79% / 56.79% | 62.21% / 62.21% | 60.84% |
| SleepEDF | 47.29% / 47.29% | 61.47% / 61.47% | 61.03% |
| SEED | 63.59% / 63.59% | 61.00% / 61.00% | 65.59% |
| EEGMAT | 70.00% / 70.00% | 83.33% / 83.33% | 93.33% |
| Nakanishi2015 | 9.44% / 9.44% | 7.78% / 7.78% | 10.00% |

The selected column uses one common step-600 checkpoint, not per-dataset best steps. Important failures remain: 2a/2b/SSVEP are near chance, P300 discrimination is weak, and some class recalls collapse. Frozen step-1200 2a recalls are 34.72%, 70.83%, 1.39%, 0%; 2b recalls are 90% and 10%. Nakanishi predicts the same semantic class for most trials and has zero reversal consistency. Conversely, frozen CHB-MIT seizure recall recovers to 85.71% at step 1200 (50% at step 600), while SEED declines from 65.59% to 61.00%. Dataset/class trends matter even when the macro score is nearly unchanged. EEGMAT still has just 30 validation windows.

Artifacts: `comparison_0800.json`, `comparison_1000.json`, `comparison_1200.json`, full `training_blocks.json`, branch metrics/validation/reload files. Use `frozen/best.pt` for the validation-selected shared model (step 600); `frozen/last.pt` is step 1200. The initial 1200-step comparison is finished; no automatic extension to 3000 or test-set evaluation was launched.

## Initial 2a instruction and input-interface audit

Upstream LEAF instruction tuning blends the task vector with the normalized mean of all candidate-label vectors in 90% of examples; its inference helper supports task-only and blended modes. Test this supported variant on the frozen initial encoder with the same 2,016 training and 288 validation trials. All 381 trainable encoder tensors and BN buffers match; only the 2a instruction-buffer row changes. Every example receives all four native labels (`Left`, `Right`, `Foot`, `Tongue`), not its true answer. Neither live JEV branch was modified.

| Readout on identical 2a validation trials | Task only | Task + all candidates |
| --- | ---: | ---: |
| Fixed C=1 linear probe | 30.21% | 30.21% |
| Source-subject-selected probe | 29.51% (C=0.01) | 32.64% (C=0.1) |
| Released output projection + native prototypes | 25.69% | 25.35% |

Both source-CV maxima are 38.00% BA. The selected-probe validation gain is nine correct examples (+3.125 points), with a different selected C, so it does not isolate an instruction-only gain. Fixed-C and released-projection readouts do not improve. This does not support missing candidate blending as the main explanation for poor 2a transfer. All probe fits converge; reloaded probe metrics, encoder immutability, identical IDs and the exact native blending formula were checked. The released projection readout is calculated in float32 from saved BF16-encoder queries, not an end-to-end BF16 forward. Evidence: `representation_probe_0200/native_candidate_control/{instruction_comparison,plan,complete}.json`, source-CV files, and saved scores/features. No test evaluation or JEV retraining is included in these figures.

A separate code/input audit finds a concrete distribution difference to investigate next: upstream LEAF's 2a builder applies 22-channel common-average reference and 0.3–40 Hz filtering, then robust scaling across a subject's epoch batch. The benchmark builder specifies MOABB 0.1–75 Hz without explicit CAR, and the current online adapter applies per-epoch robust scaling without adding CAR or 0.3–40 Hz filtering. Across all 2,016 current training trials, the median fraction of raw signal energy in the channel-common component is 80.38% (5th–95th percentiles 69.60–88.60%); the median 40–75 Hz share of 0.3–75 Hz Hann-windowed power is 3.34%. These are descriptive input measurements, not evidence that common activity is noise or that removing it improves accuracy. Code inspection also does not fully establish the historical cache's provenance. Evidence, exact formulas and inspected source hashes: `input_interface_audit.json`. A controlled input-reference comparison is a better next diagnostic than attributing the remaining failure to JEV's output slots; it was not added to the completed matched BN run.

## Startup verification

Implementation `55008f1`: 22 focused tests passed. Both one-GPU real-model smoke tests completed successfully. The frozen branch retained BN count 32,365 and mean variance 26.5335 while encoder gradients remained nonzero (norm 2.1252); the updating branch advanced BN count to 32,381 and mean variance 29.5483 (encoder gradient norm 1.9542). Both had a peak allocated memory of 58.71 GiB. These two-update checks are functional validation, not evidence of classification superiority.

Checkpoint round trips covered all 11 tasks and 38 task/class examples. Both branches saved nine named buffers and reproduced logits exactly after deliberately perturbing the head and BN state (maximum difference 0, prediction agreement 100%). Smoke metrics/reload proof files remain under the current run; consumed smoke weights were moved into the external archive. Formal training has passed update 102 in both branches with finite loss/gradients. Initial comparison checked 3,624 tensor pairs and confirmed identical trainable weights, buffers, AdamW state, scheduler, RNG states and predictions; evidence: `initial_comparison.json`. At update 102 the updating branch BN count was 32,397; the frozen branch remained 32,365 with nonzero encoder gradients, confirming the intervention. Paired update-200 validation is recorded below; early batch scores were not used to select a winner.

## Paired validation: update 600

All 9,748 sample IDs/labels match update 400 and both branches. Normal/reversed metrics were recomputed from probabilities; both 38-example checkpoint checks cover all task/classes and reproduce logits exactly. Both branches select update 600 as their current best. Dataset-macro BA is **51.4801% updating / 55.5947% frozen**, a **4.1146-point** paired difference. Changes from update 400 are +4.0763 / +0.2164 points, respectively.

| Dataset | Updating accuracy / BA | Frozen accuracy / BA | Frozen semantic consistency under reversal |
| --- | ---: | ---: | ---: |
| BNCI2014001 | 28.47% / 28.47% | 27.43% / 27.43% | 81.25% |
| BNCI2014004 | 54.17% / 54.17% | 50.83% / 50.83% | 93.33% |
| BNCI2015001 | 49.50% / 49.50% | 75.00% / 75.00% | 92.50% |
| BNCI2014008 | 57.31% / 51.29% | 44.49% / 51.15% | 77.38% |
| BNCI2014009 | 60.94% / 46.98% | 61.98% / 48.02% | 93.75% |
| CHB_MIT | 45.42% / 66.92% | 83.50% / 68.32% | 98.17% |
| TUAB | 60.25% / 60.25% | 60.84% / 60.84% | 98.54% |
| SleepEDF | 51.25% / 51.25% | 61.03% / 61.03% | 89.44% |
| SEED | 61.34% / 61.34% | 65.59% / 65.59% | 98.39% |
| EEGMAT | 86.67% / 86.67% | 93.33% / 93.33% | 100.00% |
| Nakanishi2015 | 9.44% / 9.44% | 10.00% / 10.00% | 0.00% |

Frozen BN retains the aggregate advantage, but task-level performance is mixed. SEED improves by 7.72 points and SleepEDF by 4.40; REM recall recovers from 16.38% to 38.14%. BNCI2015001 falls by 15 points because right-hand recall falls from 93% to 59%, while feet recall rises from 87% to 91%. Frozen CHB-MIT accuracy reaches 83.50%, but seizure recall declines again, from 54.76% to 50%; its BA is essentially flat at 68.32%. Because these changes occur with identical frozen BN statistics, BN drift alone cannot explain them. Shared optimization and cross-subject generalization remain candidate mechanisms, not established causes.

The frozen model's 2a/2b option consistency improves to 81.25% / 93.33% without meaningful classification recovery (27.43% / 50.83%). Frozen 2b now predicts the left semantic class for 105/120 normal-order cases; it no longer selects A in both orders for every example. This is consistent with the known-category control: option following can improve while EEG category discrimination remains weak. Nakanishi is still near chance and inconsistent under reversal.

The small macro-score improvement should not be read as broad progress. EEGMAT has only 30 validation examples: one additional correct prediction contributes +0.3030 points to the 11-dataset macro BA, exceeding the frozen model's total +0.2164-point gain. Excluding EEGMAT **for a descriptive sensitivity check only**, the other ten datasets' mean BA changes from 51.9161% to 51.8208%. The prespecified 11-dataset macro BA remains the selection rule; it has not been changed after observing results. For the updating branch, the EEGMAT rebound alone contributes +3.0303 of its +4.0763-point aggregate gain.

Evidence: [paired update-600 audit](../artifacts/benchmark/cross_subject_fold0_bn_control/comparison_0600.json), branch `validation_0600.json` and `reload_check_0600.json`. Updating BN mean running variance is 43.2360; frozen remains 26.5335. Training continues under the unchanged matched protocol.

### Training trend through update 1200

Every row below averages exactly 100 sampled training updates; these are not full-training-set evaluations. Source: `training_blocks.json` and the per-branch JSONL logs.

| Updates | Updating loss / accuracy | Frozen loss / accuracy |
| --- | ---: | ---: |
| 101–200 | 1.00035 / 46.78% | 0.95501 / 50.28% |
| 201–300 | 0.96114 / 50.07% | 0.93282 / 52.27% |
| 301–400 | 0.93779 / 52.18% | 0.91114 / 53.92% |
| 401–500 | 0.92618 / 52.83% | 0.89873 / 54.86% |
| 501–600 | 0.92264 / 53.66% | 0.88031 / 56.34% |
| 601–700 | 0.92392 / 52.98% | 0.87639 / 56.51% |
| 701–800 | 0.90687 / 54.76% | 0.86369 / 57.84% |
| 801–900 | 0.89756 / 55.48% | 0.85711 / 58.18% |
| 901–1000 | 0.89192 / 56.01% | 0.84897 / 58.28% |
| 1001–1100 | 0.89887 / 56.05% | 0.84313 / 59.61% |
| 1101–1200 | 0.89572 / 56.08% | 0.84683 / 58.87% |

Training-log integrity was checked through update 643 in both branches: all 543 updates from 101 are present, losses and gradient norms are finite, every update contains eight examples from each of the 11 datasets, and aggregate loss/accuracy agree with their dataset means. Learning rates match across branches at every update; BN counts and modes follow the specified intervention throughout. Evidence: `training_health.json` under the current run root. Over updates 501–600, median pre-clipping gradient norms are 5.20 / 4.25 (updating/frozen); both branches clip every update at the configured norm of 1. Clipping frequency alone does not establish an optimization fault, so the matched configuration is unchanged.

## Paired validation: update 400

All 9,748 validation IDs and labels match update 200 and match across branches. Metrics were recomputed from the saved per-example probabilities; both models reproduced logits exactly on the 38-example reload check covering every task/class. Both branches now select update 400 as their best saved checkpoint. Dataset-macro BA is **47.4038% updating / 55.3783% frozen**, a **7.9744-point** paired difference. From update 200, these changed by +1.1519 / +0.8514 points. This supports a continued aggregate benefit from freezing BN over these two checkpoints, not a universal per-task improvement.

| Dataset | Updating accuracy / BA | Frozen accuracy / BA | Frozen semantic consistency under reversal |
| --- | ---: | ---: | ---: |
| BNCI2014001 | 24.31% / 24.31% | 26.04% / 26.04% | 9.72% |
| BNCI2014004 | 53.33% / 53.33% | 50.00% / 50.00% | 0.00% |
| BNCI2015001 | 51.50% / 51.50% | 90.00% / 90.00% | 98.50% |
| BNCI2014008 | 45.24% / 50.34% | 47.51% / 52.08% | 73.96% |
| BNCI2014009 | 51.91% / 45.73% | 54.69% / 49.06% | 76.74% |
| CHB_MIT | 65.78% / 75.90% | 79.84% / 68.47% | 98.17% |
| TUAB | 58.25% / 58.25% | 61.23% / 61.23% | 97.75% |
| SleepEDF | 40.88% / 40.88% | 56.63% / 56.63% | 93.55% |
| SEED | 60.65% / 60.65% | 57.87% / 57.87% | 98.63% |
| EEGMAT | 53.33% / 53.33% | 90.00% / 90.00% | 96.67% |
| Nakanishi2015 | 7.22% / 7.22% | 7.78% / 7.78% | 0.00% |

The main unresolved observations are specific, not hidden by the aggregate score:

- Frozen 2b selects **slot A for all 120 examples in both option orders**, so its semantic consistency is 0% even though normal-order accuracy is 50%. Frozen 2a has zero feet/tongue recall and only 9.72% semantic consistency. Nakanishi remains near chance with zero consistency.
- Frozen CHB-MIT accuracy increased from 78.00% to 79.84%, but seizure recall decreased from 66.67% to 54.76%; BA fell from 72.87% to 68.47%. Overall accuracy alone would miss this deterioration.
- Frozen SleepEDF aggregate BA is almost unchanged (56.92% to 56.63%), while REM recall fell from 50.61% to 16.38% and N1 recall rose from 11.98% to 34.72%. Frozen SEED accuracy fell from 60.95% to 57.87%.
- Updating EEGMAT fell from 86.67% to 53.33%, with arithmetic recall dropping to 6.67%; its 30-example validation set is small. Frozen EEGMAT reached 90%, also on only 30 examples.

Thus BN freezing mitigates one failure mechanism, while class-specific drift and weak signal-to-option binding remain. Those declines also occur with fixed BN statistics, so BN drift cannot be the sole explanation. These are one-seed validation observations, not independent test results. Keep the planned matched runs unchanged while investigating the option-binding failure separately.

Training updates 301–400: updating mean loss / sampled accuracy **0.93779 / 52.1818%**; frozen **0.91114 / 53.9205%**, each over exactly 100 updates. Evidence: [paired comparison and class-recall changes](../artifacts/benchmark/cross_subject_fold0_bn_control/comparison_0400.json), branch `validation_0400.json`, `reload_check_0400.json`, and the continuously updated `metrics.jsonl` files. Updating BN mean running variance at validation was 87.6562; frozen remained 26.5335.

## Known-category option controls at update 400

The follow-up distinguishes knowing how to map an explicitly stated category to an option from obtaining that category from EEG. Select one deterministic **training** example for each of the 38 task/classes. Every class is tested at every answer position using cyclic rotations, yielding 222 cases per condition. These are deliberately answer-provided positive controls, **not EEG classification or generalization scores**. Cyclic rotations cover all positions, not every possible permutation.

| Model and input | Correct / cases |
| --- | ---: |
| Native JEV, category stated in native bare text template | 222 / 222 |
| Native JEV, category stated in EEG wrapper without signal tokens | 222 / 222 |
| Frozen-BN step 400, category stated in native bare text template | 222 / 222 |
| Frozen-BN step 400, category stated in EEG wrapper without signal tokens | 222 / 222 |
| Frozen-BN step 400, actual EEG plus stated category | 221 / 222 |

Thus the adapted model retains the ability to follow a known category and choose its corresponding letter. With actual EEG and the explicit hint, 2a, 2b and Nakanishi all pass every cyclic case. The only error is normal-EEG example `TUAB-24228`, normal order: uncalibrated choice probabilities are 0.497855 for the correct A versus 0.502145 for B. In the paired **signal-only** control, 2a gets 8/16, 2b 2/4 and Nakanishi 7/144 cases correct. These counts reuse just one source example per class; they are not full-training-set metrics.

The observed fixed-A behavior on EEG validation therefore does not establish loss of the general text-to-option mapping skill. The stronger supported interpretation is difficulty obtaining reliable category evidence from the EEG path. This control does not separate limitations of the encoder, connector and EEG-conditioned language computation. It supports focusing subsequent diagnosis on that path and its cross-subject generalization, while preserving the functioning text/choice interface.

Evidence and exact prompts: `option_binding_0400/{plan,cases,metrics,verification}.json` and `predictions.jsonl` under the current run root. All 1,332 recorded decisions were checked against their probabilities and option maps. The adapted state was strictly restored, remained unchanged throughout inference, and is preserved with its step/config in `model_state.pt`; no optimizer was created. The native template was checked against local `laya/JEV-27B-VL/README.md` and `server_vllm.py`. Both live training branches continued without source or weight changes. Helper: `scripts/diagnose_option_binding.py`.

## Paired validation: update 200

Exact same 9,748 validation IDs/labels in both branches, also matching the historical monitor subset. Dataset-macro BA: updating BN **46.2519%**, frozen BN **54.5269%** (+8.2749 percentage points). Both saved models reproduce logits exactly on all 38 task/class diagnostic examples. This is one seed and the first checkpoint; later checkpoints are still required to assess stability.

| Dataset | Updating accuracy / BA | Frozen accuracy / BA | Frozen semantic consistency under reversal |
| --- | ---: | ---: | ---: |
| BNCI2014001 | 26.04% / 26.04% | 25.35% / 25.35% | 1.74% |
| BNCI2014004 | 52.50% / 52.50% | 50.00% / 50.00% | 100.00% |
| BNCI2015001 | 50.00% / 50.00% | 81.00% / 81.00% | 81.50% |
| BNCI2014008 | 52.38% / 51.10% | 54.99% / 51.67% | 51.22% |
| BNCI2014009 | 61.81% / 43.33% | 64.24% / 44.79% | 50.00% |
| CHB_MIT | 17.92% / 55.12% | 78.00% / 72.87% | 95.93% |
| TUAB | 57.86% / 57.86% | 59.03% / 59.03% | 97.56% |
| SleepEDF | 18.29% / 18.29% | 56.92% / 56.92% | 93.01% |
| SEED | 61.19% / 61.19% | 60.95% / 60.95% | 96.53% |
| EEGMAT | 86.67% / 86.67% | 86.67% / 86.67% | 100.00% |
| Nakanishi2015 | 6.67% / 6.67% | 10.56% / 10.56% | 0.00% |

BN freezing substantially improves BNCI2015001, CHB-MIT and SleepEDF at this checkpoint, but does not resolve 2a, P300 or SSVEP. Frozen 2a accuracy is near chance and its semantic consistency is only 1.74%; Nakanishi consistency is 0%. These are unresolved answer-position/semantic-binding failures, not evidence that the full 11-task model is working. Detailed normal/reversed slot counts, confusion matrices and paired metrics are in `comparison_0200.json`. Keep the matched branches running unchanged to subsequent checks before selecting an intervention.

This is not simply a model always choosing the same letter: frozen 2a predominantly selects slot A in normal order (279/288), but slot C after reversal (278/288), changing the associated semantic class. The supported observation is strong sensitivity to the candidate arrangement. The interval from the update-200 training log to update 201 was 478.29 / 478.40 seconds for updating/frozen, including full monitor validation, checkpoint writing, reload verification and the next training update; it is not a pure inference benchmark.

Training updates 101–200: updating mean loss 1.00035 and sampled accuracy 46.7841%; frozen mean loss 0.95501 and sampled accuracy 50.2841%. Mean BN running variance ranged 33.67–913.82 for the updating branch versus a constant 26.5335 for the frozen branch. This is training-batch evidence, not full-training-set evaluation.

The next complete block, updates 201–300, has mean loss / sampled accuracy **0.96114 / 50.0682%** for updating and **0.93282 / 52.2727%** for frozen. Exactly 100 updates contribute to each mean; per-dataset values are in `training_block_0300.json`. These are training observations, not a validation result; the subsequent update-400 validation is recorded above.

## Read-only diagnostic evidence while paired training runs

`bn_update_audit.json` accounts for the 16 BN calls per optimizer update with momentum 0.1. Because the task order is fixed, only 18.53% of the prior update's running statistic remains after the next update; newly observed statistics contribute 17.78% from SleepEDF, 15.39% from SEED, 10% from Nakanishi, but 2.06% from the first task (2a). These are EMA coefficients, not measured per-task feature magnitudes or proof of a validation effect. They explain why equal example counts do not imply equal influence on shared BN statistics. The branch comparison tests the resulting train/eval normalization behavior directly.

A fixed, unfitted fundamental-frequency Fourier score on all 1,440 Nakanishi **training** rows achieved 77.0139% accuracy (random choice 8.33%). Candidate frequencies came only from the existing task catalog; a Hann window and per-channel energy normalization were fixed in advance. No validation/test access or parameter tuning. This confirms that the raw training inputs carry usable frequency/class information; it does not measure JEV performance or identify whether losses arise in preprocessing, LEAF, the connector or the language model. Results and per-example scores: `nakanishi_frequency_train.json`. Reproduce with `scripts/audit_benchmark.py --frequency-check-train --out <path>`; the existing audit entry point was extended, not a separate experimental model. A synthetic check covers all 12 frequencies under phase, channel gain and offset changes.

Follow-up using the **exact current online preprocessing** (`SignalStore.batch`, including robust normalization, 65-channel mapping and padding to 900 samples) on the identical 1,440 training IDs achieved 73.9583% with the same fixed Fourier rule, versus 77.0139% on raw cached inputs. Thus substantial class-related frequency information remains after input adaptation; this does not identify which downstream representation/decoder component causes poor JEV accuracy. The Fourier aggregation now covers 65 rather than 8 channels and a padded window, so the difference is a whole-input-pipeline comparison, not an isolated estimate of interpolation loss. Evidence: `nakanishi_frequency_online_train.json`; reproduce with `--frequency-check-train --online-input`. No validation/test data were used and neither training branch was changed.

A CPU-only input diagnostic sampled the exact training calls at updates 101, 120 and 140 with the **fixed released tokenizer CNN**, before BN. Mean activation variance was about 7–34 for most tasks, 74.73 for Nakanishi and 354.89 for SEED; one two-example SEED call reached 1,134.08. These descriptive samples are not a population estimate. Reconstructing BN's EMA from these fixed-CNN activation variances predicted the live updating branch's mean running variance as 40.3964 / 78.6547 / 119.5143, versus observed 40.3991 / 78.6530 / 119.4984. Relative errors were all <0.014%, consistent with input-driven running-statistic movement rather than substantial CNN weight changes at these early updates. This supports the normalization mechanism but does not establish its effect on held-out accuracy. Evidence/reproduction: `bn_input_train.json`, `scripts/audit_benchmark.py --bn-input-check-train --out <run>/bn_input_train.json`. This diagnostic used CPU and training samples only, with no change to either active training branch.

## Frozen LEAF query diagnostic at update 200

The frozen-BN branch's saved encoder was restored with all five encoder buffers and kept in evaluation mode. Its eight query tokens were flattened to 2,048 features. Separate, fixed `StandardScaler` + balanced logistic-regression pipelines were fit on all training rows for three unresolved datasets. No hyperparameter search or test access. All 1,044 validation IDs and labels exactly match the corresponding JEV update-200 predictions.

| Dataset | Linear probe train accuracy / BA | Linear probe validation accuracy / BA | JEV validation accuracy / BA |
| --- | ---: | ---: | ---: |
| BNCI2014001 (2a) | 97.72% / 97.72% | 28.47% / 28.47% | 25.35% / 25.35% |
| BNCI2014009 (P300) | 90.32% / 92.53% | 73.44% / 54.48% | 64.24% / 44.79% |
| Nakanishi2015 | 99.38% / 99.38% | 11.11% / 11.11% | 10.56% / 10.56% |

This does **not** support attributing all poor performance to the JEV interface: 2a and Nakanishi also generalize poorly with this direct linear readout despite near-perfect training fit. P300 has somewhat more accessible held-out information, but its BA remains modest. These are results for one fixed regularization setting and one validation subject per dataset; they do not prove that the encoder cannot represent the tasks. The 2,048-dimensional classifier may overfit, and its optimization/data exposure differ from JEV. By update 200, the balanced JEV sampler had visited 1,110/2,016 distinct 2a rows, 1,203/4,608 P300 rows and 989/1,440 Nakanishi rows; the probes use all training rows. Do not interpret this as a matched training-recipe comparison.

The same previously fixed Fourier rule was subsequently applied to the 180 Nakanishi validation rows, with no parameter change. Accuracy was **12.78% on the raw cache and 65.56% after exact online preprocessing**, compared with 11.11% for the LEAF-query probe. All IDs/labels match. The processed inputs therefore retain useful frequency information on this held-out subject. The raw-to-processed change includes robust clipping, spatial interpolation and padding; it is not an isolated estimate of any single preprocessing effect. This result motivates checking the representation/readout's cross-subject frequency generalization, without asserting an encoder implementation bug.

Evidence: [diagnostic metrics](../artifacts/benchmark/cross_subject_fold0_bn_control/representation_probe_0200/metrics.json), `comparison.json`, `sampling_0200.json`, `frequency_{raw,online}_validation.json`, and `verification.json` in the same directory. The exact encoder snapshot, features, fitted probes and per-example predictions are preserved there. Encoder parameters and buffers remained unchanged; all 9,108 saved train/validation probabilities reproduce exactly after reloading the fitted probes. All three fits converged without warnings. The frequency-score synthetic test passed. Neither active BN branch was changed.

### Source-subject regularization selection and initial-encoder control

The follow-up selects C using only leave-one-training-subject-out BA, with the scaler refit inside every fold. Choices for all datasets were saved before accessing outer validation arrays. The six prespecified candidates were `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]`. The same procedure was repeated with the step-100 encoder, before unfreezing. All 381 trainable encoder tensors and four released-model buffers in that initial snapshot exactly match the release; the fifth buffer is the current task-instruction table. Every sample ID and label matches the step-200 diagnostic.

| Dataset | Initial C | Initial validation accuracy / BA | Step-200 C | Step-200 validation accuracy / BA |
| --- | ---: | ---: | ---: | ---: |
| BNCI2014001 | 0.01 | 29.51% / 29.51% | 0.01 | 29.86% / 29.86% |
| BNCI2014009 | 0.01 | 76.22% / 61.56% | 0.1 | 73.96% / 58.12% |
| Nakanishi2015 | 0.001 | 10.56% / 10.56% | 0.001 | 10.56% / 10.56% |

For the step-200 encoder, selected-C training accuracy is 71.33% / 83.18% / 26.67%, respectively. Stronger regularization reduces the near-perfect training fit but does not recover 2a or Nakanishi validation performance. Initial and step-200 selected-C source-subject CV BA are 38.00% / 37.85% for 2a, 62.02% / 62.07% for P300 and 11.67% / 12.01% for Nakanishi. At fixed C=1, initial validation BA is 30.21% / 53.85% / 11.11%, versus step-200 28.47% / 54.48% / 11.11%. Thus these low 2a/SSVEP scores were already present in the initial query representation under this readout; this is not evidence of a new collapse caused solely by the first 100 encoder updates. The P300 selected-C comparison changes C as well as encoder weights, so it does not isolate an encoder-only effect.

There were 138 inner-fold fits per encoder, all without convergence warnings. Reloaded probe probabilities match exactly on all saved examples. Evidence: `representation_probe_0200/source_cv_comparison.json`, `encoder_comparison.json`, and the `source_cv/{plan,selection,metrics,verification}.json` files under both the diagnostic root and `initial_encoder_control/`. The fixed encoder had already seen all source subjects during its training history; these inner scores evaluate the readout, not a fully held-out end-to-end model. Outer validation has also been inspected repeatedly. No independent test claim is made. The matched BN training branches remain unchanged; their subsequent update-400 comparison is recorded above.

## Checkpoint correction and initialization

The previous trainer saved trainable parameters but omitted LEAF SyncBatchNorm running statistics. Online in-process validation scores remain evidence for the live models, but later reloads cannot reproduce those models exactly. This implementation now saves all buffers, optimizer/scheduler and RNG state; unit tests require identical restored predictions, next updates and random draws. Real-model round-trip checks run on held-out examples for every task/class.

The interrupted cosine run's last saved update is 100, during the encoder freeze. All 381 saved trainable encoder tensors exactly matched the released LEAF weights, so its original BN statistics are recoverable. Both branches use the same new seed 142 because the old checkpoint did not save RNG state. This is a matched branch initialization, not an exact continuation of the original random stream. Source run logs and old artifacts are retained in the external archive; compact metrics are in `reference_cosine_partial/`.

## Retained constant-LR control

Archived implementation: `d11b0bf`. User stopped training on 2026-10-24 at logged step 1,087. Preserved archived `last.pt`: step 1,000; archived `best.pt`: step 600, dataset-macro BA 50.0512%. No final full validation/test was run. Compact metrics/predictions remain under `artifacts/benchmark/reference_constant_lr/`. The prior 100-update encoder freeze was not LR warmup; rates were constant.

Dataset-macro BA at updates 200/400/600/800/1000: 45.3003% / 48.2431% / 50.0512% / 49.6711% / 46.0625%. All monitor sample IDs and labels are identical across these checkpoints.

| Dataset | Step 600 accuracy / BA | Step 800 accuracy / BA | Step 1000 accuracy / BA |
| --- | ---: | ---: | ---: |
| BNCI2014001 | 27.08% / 27.08% | 28.12% / 28.12% | 26.39% / 26.39% |
| BNCI2014004 | 55.00% / 55.00% | 48.33% / 48.33% | 50.00% / 50.00% |
| BNCI2015001 | 49.00% / 49.00% | 77.50% / 77.50% | 35.00% / 35.00% |
| BNCI2014008 | 52.32% / 50.24% | 54.18% / 50.19% | 58.12% / 49.60% |
| BNCI2014009 | 61.11% / 48.33% | 74.48% / 48.85% | 82.47% / 49.48% |
| CHB_MIT | 71.89% / 72.76% | 77.60% / 66.17% | 84.93% / 60.47% |
| TUAB | 60.50% / 60.50% | 59.96% / 59.96% | 60.69% / 60.69% |
| SleepEDF | 43.47% / 43.47% | 47.92% / 47.92% | 53.55% / 53.55% |
| SEED | 62.51% / 62.51% | 62.66% / 62.66% | 63.73% / 63.73% |
| EEGMAT | 73.33% / 73.33% | 50.00% / 50.00% | 50.00% / 50.00% |
| Nakanishi2015 | 8.33% / 8.33% | 6.67% / 6.67% | 7.78% / 7.78% |

CHB-MIT seizure recall fell from 73.81% at step 600 to 30.95% at step 1000 despite rising overall accuracy. BNCI2014009 target recall was zero at step 1000. Learning-rate scheduling is one candidate explanation to test; these observations do not establish its causal role. Training accuracy in the JSONL is sampled, per-update accuracy, not fixed-checkpoint full-training-set evaluation.

## Unchanged fold-0 data

476,032 training epochs, 58,362 full validation epochs and 60,275 test epochs. One joint subject-disjoint fold; not full benchmark LOSO/10-fold evaluation. Released LEAF pretraining exposure is not excluded. Source: `datasets/benchmark/signals/fold0/complete.json`.

| Dataset | Train | Full validation | Test |
| --- | ---: | ---: | ---: |
| BNCI2014001 | 2016 | 288 | 288 |
| BNCI2014004 | 1120 | 120 | 160 |
| BNCI2015001 | 2000 | 200 | 200 |
| BNCI2014008 | 25200 | 4200 | 4200 |
| BNCI2014009 | 4608 | 576 | 576 |
| CHB_MIT | 23994 | 491 | 1807 |
| TUAB | 43218 | 5148 | 5508 |
| SleepEDF | 327294 | 43735 | 43932 |
| SEED | 44122 | 3394 | 3394 |
| EEGMAT | 1020 | 30 | 30 |
| Nakanishi2015 | 1440 | 180 | 180 |

Earlier approaches and their reports are archived in Git (including `d11b0bf`); ignored datasets, environments, logs and checkpoints have not been deleted.
