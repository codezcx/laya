# Online EEG → text-conditioned LEAF → JEV: joint cross-subject fold

Current implementation trains one shared model on 11 classification datasets from EEG-FM-Benchmark. The current experiment tests updating versus fixed BatchNorm statistics under the same warmup/cosine schedule. Task catalog: `configs/benchmark_tasks.json`; training settings: `configs/benchmark_train.json`. Ignored datasets, checkpoints and logs are retained. No frozen-query cache is used by this run.

Status: both matched branches completed the planned step-1200 diagnostic stop on 2026-10-25. Best validation macro BA is 51.85% for updating BN (step 1200) and 55.59% for frozen BN (step 600). The 3000-step setting remains the LR horizon, not a claim that 3000 updates were run. No test evaluation or further training was started automatically. See [RESULTS.md](RESULTS.md) for all six validations and the saved checkpoint audit.

## Cross-subject protocol

Start with joint fold0. For ordinary datasets, sort actual subject IDs numerically, hold out one subject for test and the next subject for validation; all other subjects train. For TUAB and SleepEDF, shuffle subjects with seed42 and partition into10groups; hold out group0 for test, group1 for validation, and train on8groups. chb01 and chb21 are one person and always share a split, per [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

This run is ONE synchronized joint cross-subject fold. It is NOT the complete benchmark's independently trained per-dataset LOSO/10-fold result. The additional validation subject/group reduces training data relative to a simple train/test LOSO fold. The code supports choosing other valid fold indices, but the launcher currently starts only fold0; no claim of completed cross-validation. Do not combine repetitions of the same held-out subject as independent folds.

No target-test subject contributes to SFT. Validation subjects do not contribute to SFT either. Cache usage flags encode a different within-subject few-shot protocol and are not used. TUAB source cache combines original train/eval data; our subject-group partition differs from the official TUAB split. Released LEAF pretraining exposure is not ruled out by this downstream protocol.

Eleven datasets: BNCI2014001 (left/right/feet/tongue), BNCI2014004 (left/right), BNCI2015001 (right/feet), BNCI2014008 and BNCI2014009 (target/non-target), CHB-MIT (non-seizure/seizure), TUAB (normal/abnormal), SleepEDF (wake/N1/N2/N3/REM), SEED (negative/neutral/positive), EEGMAT (rest/arithmetic), Nakanishi2015 (12frequencies). SEED-VIG regression and ThingsEEG2 matching remain pending output-protocol clarification and data availability.

## Loading raw signals

`scripts/prepare_benchmark.py` exports original arrays to float32 memory-mapped files. This is CPU data serialization, NOT feature extraction. Original channel layout and sampling rate are retained. All-zero/nonfinite windows are excluded with original indices/reasons; CHB audit found1,144all-zero windows among27,436. Source rows are assigned to splits only after export. Every valid training window remains available; no dataset is capped.

`eeg_align/signal_data.py` verifies manifest hashes, asserts disjoint subject sets, and loads only requested epochs from disk. It groups a microbatch by dataset to apply the correct sampling rate, channel mapping and task instruction. Nakanishi exported epochs have829/830/831samples; pad0–2samples without cropping. Other sources retain their full epoch durations.

Online preprocessing: polyphase resampling to200Hz, per-epoch robust percentile clipping and median/IQR scaling; use standard deviation only when a sparse signal has zero IQR. Map to LEAF's65-channel template with released aliases/interpolation. Exclude TUAB A1/A2 references absent from the template, retaining19scalp channels. SEED clips to[-10,10] after mapping. Sparse montages entail extensive interpolation; this is an input adaptation limitation.

CHB-MIT/SleepEDF bipolar leads use a minimum-norm incidence-matrix inverse with arbitrary component references, NOT recovered physical monopolar EEG. Missing zero-filled leads are excluded from constraints. Full CHB audit found up to3.04%relative reconstruction error; enforce5%maximum. Keep full30-second sleep epochs as three ordered10-second segments; other epochs fit one segment. Pad incomplete100-sample tokenizer patches instead of dropping tails.

## Model and objective

Released LEAF plus native `laya/JEV-27B-VL`. LEAF FiLM/QFormer is conditioned on task text only; true labels never enter its input. Each <=10-second segment emits 8 final-normalized 256-dimensional query tokens (24 for 30-second sleep epochs). Project 256→5120, concatenate with question/option embeddings, and apply JEV's native 16-position choice head. Random option permutation and answer-position cross entropy are unchanged; unused choices are masked.

LEAF is frozen through update 100, then its tokenizer, Transformer branches and QFormer are trained. Native Qwen base weights stay frozen; the original JEV LoRA is always active. Unused LEAF reconstruction heads, mask token and text-output projection remain frozen. The optimizer is AdamW with weight decay 0.01 and gradient norm clipping at 1.

## Current matched BatchNorm experiment

Two independent single-H200 branches run concurrently. Both start from the same saved update-100 checkpoint of the interrupted cosine run. All 381 trainable LEAF tensors were checked equal to the released checkpoint; because its encoder was never unfrozen, original BN buffers can be recovered exactly. Its old RNG state was not saved, so both branches explicitly reset Python, NumPy, CPU/CUDA torch RNGs to branch seed 142. This is a matched new continuation, not bitwise continuation of the original run.

- `updating`: ordinary training behavior, BatchNorm uses current minibatch statistics and updates running statistics.
- `frozen`: BatchNorm uses the released running mean/variance during training and evaluation. Its affine weight/bias and all other active encoder parameters still receive gradients. Other dropout remains active.

Both branches use world size 1, global batch 88, microbatch 22 (four accumulation passes), eval microbatch 32, identical sample IDs and option permutations. Each step samples 8 examples per dataset with class/subject balancing and replacement. The only branch setting difference is BN policy. Compared with historical runs, world-size/microbatch and stochastic execution differ; historical scores are context, not the matched BN control.

Peak LRs are connector 1e-4, JEV LoRA 5e-6, choice head 1e-5, encoder 1e-6. Scheduler retains a 3,000-update horizon, 90-update linear warmup then cosine decay to 2% of peak. Encoder is zero through update 100, warms up independently at 101–190, then decays. Initial diagnostic stage stops at update 1,200; it does not compress the LR schedule to 1,200. Select later improvements only after examining the paired validation results.

## Complete state and validation

Version-2 checkpoints save all trainable named parameters, all model buffers (including BN running mean/variance/count), AdamW state, scheduler state, and per-rank Python/NumPy/torch RNG state. Frozen pretrained parameter weights remain supplied by the recorded releases. Missing-buffer legacy checkpoints are rejected except for explicitly verified frozen-warmup recovery. Post-unfreeze legacy checkpoints cannot silently borrow release statistics.

Each branch writes `initial.pt` and fingerprints its parameters/buffers. The controller verifies that both initial states match. A reload test destroys BN statistics and changes a head parameter, restores the checkpoint, and requires bitwise-identical logits on fixed held-out examples covering every dataset/class. This test runs at initialization and every validation checkpoint. Unit tests additionally require identical next optimizer updates and RNG draws after restoration.

Validation every 200 global updates uses the same fixed 9,748 rows as earlier runs (cap 2,048/dataset). Report normal/reversed option accuracy, balanced accuracy, per-subject metrics, confusion matrices and semantic consistency. Save last weights every 100 updates, plus initial/first/final and validation-selected best. No test-set selection or automatic test evaluation in this diagnostic stage.

`monitor.json` stores each branch's per-dataset 100-update training averages, full validation history and alerts. `LIVE.md` provides the latest human-readable summary. Flags include zero class recall, >95% predictions in one class, or >=10-point BA decline from the previous validation; these trigger investigation, not an automatic claim of cause. Nonfinite losses/gradients, failed reload checks or mismatched starting states stop the run. Training batch accuracy is not fixed-checkpoint full-training-set accuracy.

## Entry points, outputs and archival

Run `laya/.venv-eeg/bin/python scripts/run_benchmark.py`. The controller runs two independent one-GPU smoke tests, then the two branches from the common checkpoint. The smoke tests shorten freezing/scheduling to two updates and cover all tasks/options; their scores are not scientific results. Existing output directories are never overwritten. Source hashes prevent edits to the active implementation between stages and during monitoring. SIGTERM requests a checkpoint at the next update boundary.

- Current root: `artifacts/benchmark/cross_subject_fold0_bn_control/`.
- Initial checkpoint: `common/warmup_step0100.pt`.
- Branch metrics: `{updating,frozen}/metrics.jsonl`.
- Branch weights: `initial.pt`, `last.pt`, `best.pt` within each branch.
- Controller: `pipeline.json`; summaries: `monitor.json`, `LIVE.md`.
- Compact historical comparisons: `artifacts/benchmark/reference_constant_lr/`, `reference_cosine_partial/`.
- Shared raw signals: `datasets/benchmark/signals/`; shared MPNet/instructions remain unchanged.

## Frozen representation diagnostic

While the matched branches run, `scripts/diagnose_representation.py` freezes the saved frozen-BN step-200 LEAF encoder, including its BN buffers, and extracts the exact eight query tokens used as JEV input. Flattening produces 2,048 features per example. Three unresolved datasets are checked: BNCI2014001, BNCI2014009 and Nakanishi2015. Use all their existing training rows and the same held-out validation subjects; no test access or split changes.

Each task gets a separate fixed `StandardScaler` + `LogisticRegression(C=1, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)` pipeline. Scaling and classifier fitting use training rows only. No validation hyperparameter search. Save train/validation accuracy, BA, confusion matrices, convergence warnings, IDs, probabilities, features and the exact encoder snapshot under `cross_subject_fold0_bn_control/representation_probe_0200/`. Verify encoder parameters and buffers remain unchanged after extraction.

This is a diagnostic of accessible class information, not a matched comparison of optimizers or training exposure. A better linear-probe result would support usable information in the encoder output, but would not by itself locate a specific JEV interface bug. The extra encoder-only inference process may affect one branch's wall time; it does not update either training model.

The fixed frequency positive control can also be applied to validation with `scripts/audit_benchmark.py --frequency-check-validation [--online-input] --out <path>`. It uses the same candidate fundamentals, Hann window and per-channel energy normalization fixed for the training-only check. This is a validation diagnostic, not test evaluation or hyperparameter selection.

To check whether the fixed-C probe's generalization gap is explained by weak regularization, the same helper accepts `--source-cv-cache <representation_probe_0200> --out <representation_probe_0200/source_cv>`. On CPU, it selects C from `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]` using leave-one-training-subject-out BA, averaged equally across subjects. Refit the scaler and classifier inside each fold; break ties toward stronger regularization. Save all selections before reading outer validation arrays, then refit each selected model on all source rows and evaluate the fixed validation set once. No test data or JEV updates. The encoder was already trained using all source subjects, so inner-fold scores diagnose the readout and are not unbiased end-to-end unseen-subject estimates. This follow-up also does not turn the repeatedly inspected outer validation set into independent test evidence.

A matched initial-encoder control uses `frozen/initial.pt` (step 100, before any encoder update) with the identical extraction, fixed-C and source-subject-CV procedures. Its encoder matches the release weights and original BN statistics. Store it under `representation_probe_0200/initial_encoder_control/`; this is a diagnostic control within the current experiment, not a new training approach. It separates limitations already present in the initial query representation from changes after joint fine-tuning.

The bounded 2a instruction control uses `--native-candidate-instruction --datasets BNCI2014001 --expected-step 100` with the same initial checkpoint. Following upstream `b_instruct_tuning.py` / `c_inference.py`, mix the task vector with the normalized mean of the four cached native candidate vectors (`Left`, `Right`, `Foot`, `Tongue`), then normalize again. Every example receives the complete candidate set; its true class is never used for conditioning. Keep signal preprocessing, all encoder weights, BN statistics, train/validation IDs and probe recipes fixed. Compare with the already extracted task-only initial queries using fixed-C probes, the same source-subject C selection and the released output projection/prototype scores. Save under `representation_probe_0200/native_candidate_control/`. This checks a supported upstream input variant; it does not assume task-only conditioning is a bug or change either live training branch.

## Known-answer option-binding diagnostic

`scripts/diagnose_option_binding.py` checks all 38 task/classes with cyclic option rotations (222 cases per condition). Select exactly one deterministic **training** example per class. In the oracle conditions, the true category is deliberately stated in text, without revealing its letter. These scores measure following known categories and option positions, never EEG accuracy or generalization.

First evaluate the native JEV adapter/head on its documented bare text template and on the EEG wrapper with no signal tokens. Then restore the frozen-BN step-400 model and repeat those two controls, plus actual EEG without a hint and actual EEG with the known-category hint. The signal-only control also has only 38 distinct source examples and is not a full-training-set metric. All six conditions use the same label-independent cyclic orders, covering every answer position for every class. Positive shared temperature calibration is omitted because it cannot change argmax.

This diagnostic uses a separate inference process, no optimizer, and does not mutate either live model. Require at least 65 GiB of spare GPU memory before loading it. Save prompts, probabilities, the restored model state, and state-immutability checks under `cross_subject_fold0_bn_control/option_binding_0400/`. Its GPU work may affect a branch's wall time but does not alter that branch's training semantics.

Superseded tracked implementations are in Git (`521f865` before this replacement). Old artifacts are moved outside the repository, not deleted. See [ARCHIVE.md](ARCHIVE.md) for the inventory. Unrelated ECG-R1 working files, shared models, datasets, environments and third-party installations are retained.
