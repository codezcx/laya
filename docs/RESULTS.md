# Joint cross-subject BatchNorm comparison

Current experiment: `artifacts/benchmark/cross_subject_fold0_bn_control/`. Two single-H200 branches compare updating vs fixed BN statistics from the same update-100 checkpoint; both unfreeze LEAF parameters and keep JEV LoRA active. Initial diagnostic stop: step 1200, with unchanged 3000-step LR schedule. See [PROTOCOL.md](PROTOCOL.md). First paired validation at update 200 is complete; both checkpoint reload checks passed.

- Live summaries: `LIVE.md`, `monitor.json` under the current root.
- Per-step metrics: `{updating,frozen}/metrics.jsonl`.
- Fixed monitor validation: 9,748 rows, every 200 global updates.
- Last checkpoint: every 100 updates; best: validation-selected dataset-macro BA.
- Historical artifact index: [ARCHIVE.md](ARCHIVE.md).

## Startup verification

Implementation `55008f1`: 22 focused tests passed. Both one-GPU real-model smoke tests completed successfully. The frozen branch retained BN count 32,365 and mean variance 26.5335 while encoder gradients remained nonzero (norm 2.1252); the updating branch advanced BN count to 32,381 and mean variance 29.5483 (encoder gradient norm 1.9542). Both had a peak allocated memory of 58.71 GiB. These two-update checks are functional validation, not evidence of classification superiority.

Checkpoint round trips covered all 11 tasks and 38 task/class examples. Both branches saved nine named buffers and reproduced logits exactly after deliberately perturbing the head and BN state (maximum difference 0, prediction agreement 100%). Smoke metrics/reload proof files remain under the current run; consumed smoke weights were moved into the external archive. Formal training has passed update 102 in both branches with finite loss/gradients. Initial comparison checked 3,624 tensor pairs and confirmed identical trainable weights, buffers, AdamW state, scheduler, RNG states and predictions; evidence: `initial_comparison.json`. At update 102 the updating branch BN count was 32,397; the frozen branch remained 32,365 with nonzero encoder gradients, confirming the intervention. Paired update-200 validation is recorded below; early batch scores were not used to select a winner.

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

The next complete block, updates 201–300, has mean loss / sampled accuracy **0.96114 / 50.0682%** for updating and **0.93282 / 52.2727%** for frozen. Exactly 100 updates contribute to each mean; per-dataset values are in `training_block_0300.json`. These are new training observations, not a new validation result. The next fixed validation is update 400.

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

There were 138 inner-fold fits per encoder, all without convergence warnings. Reloaded probe probabilities match exactly on all saved examples. Evidence: `representation_probe_0200/source_cv_comparison.json`, `encoder_comparison.json`, and the `source_cv/{plan,selection,metrics,verification}.json` files under both the diagnostic root and `initial_encoder_control/`. The fixed encoder had already seen all source subjects during its training history; these inner scores evaluate the readout, not a fully held-out end-to-end model. Outer validation has also been inspected repeatedly. No independent test claim is made. The matched BN training branches remain unchanged; update 400 will test whether the normalization benefit persists.

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
