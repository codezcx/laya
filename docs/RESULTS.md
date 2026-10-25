# Joint cross-subject learning-rate experiment

Current run: `artifacts/benchmark/cross_subject_fold0_lr_cosine/`. Launched on 2026-10-25 on two H200 GPUs, fresh initialization with 90-update LR warmup and cosine decay to 2% of peak; the LEAF encoder remains frozen for the first 100 updates and warms up separately for updates 101–190. Same initialization, seed, data, sampler and global batch 88 as the retained constant-LR control. Implementation commit: `441a8fa`. See [protocol](PROTOCOL.md) for the world-size/microbatch reproducibility limitation. No new formal validation result yet.

- Status: `artifacts/benchmark/cross_subject_fold0_lr_cosine/pipeline.json`.
- Per-step loss, sampled training accuracy and learning rates: `artifacts/benchmark/cross_subject_fold0_lr_cosine/jev/metrics.jsonl`.
- Validation every 200 updates on the same 9,748 rows; last checkpoint every 100 updates.
- Planned 3,000 updates, followed by full validation/test using the best validation-selected checkpoint.

Pre-launch checks: 18 focused tests passed (schedule endpoints and delayed encoder warmup, optimizer/scheduler restoration, distributed accumulation, data adapters and choice masking). Read-only comparison confirmed all 9,748 monitor validation IDs/labels match the old run, 476,032 training epochs remain available, and two-/four-rank partitioning preserves the same 88 global sample IDs with eight per dataset per step. The two-GPU smoke test passed with production microbatch 22/rank, both option orders and all 11 datasets: encoder gradient norm 0 while frozen and 6.8471 after unfreezing, with the tokenizer parameter-update assertions passing. Peak rank-0 allocated GPU memory was 59.23 GiB; smoke execution took 91.83 seconds excluding model loading. Smoke uses a shortened two-update LR schedule and does not count as benchmark validation. The controller has advanced to formal production SFT.

## Retained constant-LR control

Archived implementation: `d11b0bf`. User stopped training on 2026-10-24 at logged step 1,087. Preserved `last.pt`: step 1,000; `best.pt`: step 600, dataset-macro BA 50.0512%. No final full validation/test was run. Artifacts remain under `artifacts/benchmark/cross_subject_fold0/`. The prior 100-update encoder freeze was not LR warmup; rates were constant.

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

## Implementation audit: BatchNorm state (2026-10-25)

Read-only audit found a checkpoint correctness omission shared by both online runs: LEAF's tokenizer contains `SyncBatchNorm` with `running_mean`, `running_var`, and `num_batches_tracked` buffers, while `train_joint.py` saves/restores only trainable named parameters. The preserved step-600 checkpoint contains this layer's affine weight/bias but none of the running-statistic buffers. Fresh checkpoint evaluation consequently restores the released LEAF statistics rather than the statistics used by the in-memory model. This does not invalidate the recorded in-process validation scores or explain their fluctuations by itself, but reloaded-checkpoint evaluation is not an exact reproduction until buffer handling is corrected. The live run's training implementation has not been modified during this audit.

Separate untested hypothesis: after encoder unfreezing, shared BatchNorm running statistics are updated on successive task-specific microbatches, while validation uses their accumulated values. These updates are not controlled by the learning-rate scheduler. Microbatch changes can affect them: the old four-rank/microbatch-4 layout processes eight CHB windows in one synchronized BN call per step; the new two-rank/microbatch-22 layout splits CHB into two calls of four global windows. Therefore even equal global sample IDs do not isolate the LR effect from BatchNorm behavior. An exact attribution needs a matched execution configuration and a separate BN-statistics control. Checkpoint buffer preservation and an in-memory versus reload prediction check are required before reporting final checkpoint-based results.

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
