# Joint cross-subject BatchNorm comparison

Current experiment: `artifacts/benchmark/cross_subject_fold0_bn_control/`. Two single-H200 branches compare updating vs fixed BN statistics from the same update-100 checkpoint; both unfreeze LEAF parameters and keep JEV LoRA active. Initial diagnostic stop: step 1200, with unchanged 3000-step LR schedule. See [PROTOCOL.md](PROTOCOL.md). No new paired validation result yet.

- Live summaries: `LIVE.md`, `monitor.json` under the current root.
- Per-step metrics: `{updating,frozen}/metrics.jsonl`.
- Fixed monitor validation: 9,748 rows, every 200 global updates.
- Last checkpoint: every 100 updates; best: validation-selected dataset-macro BA.
- Historical artifact index: [ARCHIVE.md](ARCHIVE.md).

## Startup verification

Implementation `55008f1`: 22 focused tests passed. Both one-GPU real-model smoke tests completed successfully. The frozen branch retained BN count 32,365 and mean variance 26.5335 while encoder gradients remained nonzero (norm 2.1252); the updating branch advanced BN count to 32,381 and mean variance 29.5483 (encoder gradient norm 1.9542). Both had a peak allocated memory of 58.71 GiB. These two-update checks are functional validation, not evidence of classification superiority.

Checkpoint round trips covered all 11 tasks and 38 task/class examples. Both branches saved nine named buffers and reproduced logits exactly after deliberately perturbing the head and BN state (maximum difference 0, prediction agreement 100%). Smoke metrics/reload proof files remain under the current run; consumed smoke weights were moved into the external archive. Formal training has passed update 102 in both branches with finite loss/gradients. Initial comparison checked 3,624 tensor pairs and confirmed identical trainable weights, buffers, AdamW state, scheduler, RNG states and predictions; evidence: `initial_comparison.json`. At update 102 the updating branch BN count was 32,397; the frozen branch remained 32,365 with nonzero encoder gradients, confirming the intervention. Paired validation is pending update 200; early batch scores are not used to select a winner.

## Read-only diagnostic evidence while paired training runs

`bn_update_audit.json` accounts for the 16 BN calls per optimizer update with momentum 0.1. Because the task order is fixed, only 18.53% of the prior update's running statistic remains after the next update; newly observed statistics contribute 17.78% from SleepEDF, 15.39% from SEED, 10% from Nakanishi, but 2.06% from the first task (2a). These are EMA coefficients, not measured per-task feature magnitudes or proof of a validation effect. They explain why equal example counts do not imply equal influence on shared BN statistics. The branch comparison tests the resulting train/eval normalization behavior directly.

A fixed, unfitted fundamental-frequency Fourier score on all 1,440 Nakanishi **training** rows achieved 77.0139% accuracy (random choice 8.33%). Candidate frequencies came only from the existing task catalog; a Hann window and per-channel energy normalization were fixed in advance. No validation/test access or parameter tuning. This confirms that the raw training inputs carry usable frequency/class information; it does not measure JEV performance or identify whether losses arise in preprocessing, LEAF, the connector or the language model. Results and per-example scores: `nakanishi_frequency_train.json`. Reproduce with `scripts/audit_benchmark.py --frequency-check-train --out <path>`; the existing audit entry point was extended, not a separate experimental model. A synthetic check covers all 12 frequencies under phase, channel gain and offset changes.

Follow-up using the **exact current online preprocessing** (`SignalStore.batch`, including robust normalization, 65-channel mapping and padding to 900 samples) on the identical 1,440 training IDs achieved 73.9583% with the same fixed Fourier rule, versus 77.0139% on raw cached inputs. Thus substantial class-related frequency information remains after input adaptation; this does not identify which downstream representation/decoder component causes poor JEV accuracy. The Fourier aggregation now covers 65 rather than 8 channels and a padded window, so the difference is a whole-input-pipeline comparison, not an isolated estimate of interpolation loss. Evidence: `nakanishi_frequency_online_train.json`; reproduce with `--frequency-check-train --online-input`. No validation/test data were used and neither training branch was changed.

A CPU-only input diagnostic sampled the exact training calls at updates 101, 120 and 140 with the **fixed released tokenizer CNN**, before BN. Mean activation variance was about 7–34 for most tasks, 74.73 for Nakanishi and 354.89 for SEED; one two-example SEED call reached 1,134.08. These descriptive samples are not a population estimate. Reconstructing BN's EMA from these fixed-CNN activation variances predicted the live updating branch's mean running variance as 40.3964 / 78.6547 / 119.5143, versus observed 40.3991 / 78.6530 / 119.4984. Relative errors were all <0.014%, consistent with input-driven running-statistic movement rather than substantial CNN weight changes at these early updates. This supports the normalization mechanism but does not establish its effect on held-out accuracy. Evidence/reproduction: `bn_input_train.json`, `scripts/audit_benchmark.py --bn-input-check-train --out <run>/bn_input_train.json`. This diagnostic used CPU and training samples only, with no change to either active training branch.

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
