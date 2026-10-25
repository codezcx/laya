# Joint cross-subject BatchNorm comparison

Current experiment: `artifacts/benchmark/cross_subject_fold0_bn_control/`. Two single-H200 branches compare updating vs fixed BN statistics from the same update-100 checkpoint; both unfreeze LEAF parameters and keep JEV LoRA active. Initial diagnostic stop: step 1200, with unchanged 3000-step LR schedule. See [PROTOCOL.md](PROTOCOL.md). No new paired validation result yet.

- Live summaries: `LIVE.md`, `monitor.json` under the current root.
- Per-step metrics: `{updating,frozen}/metrics.jsonl`.
- Fixed monitor validation: 9,748 rows, every 200 global updates.
- Last checkpoint: every 100 updates; best: validation-selected dataset-macro BA.
- Historical artifact index: [ARCHIVE.md](ARCHIVE.md).

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
