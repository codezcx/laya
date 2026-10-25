# Online EEG → text-conditioned LEAF → JEV: joint cross-subject fold

Current implementation trains one shared model on 11 classification datasets from EEG-FM-Benchmark. The current experiment compares the existing 2a input adapter with raw LEAF preprocessing, using the same step-600 checkpoint and frozen BN statistics. Task catalog: `configs/benchmark_tasks.json`; training settings: `configs/benchmark_train.json`. Ignored datasets, checkpoints and logs are retained. No frozen-query cache is used by this run.

The prior BN comparison completed step 1200; its best frozen-BN checkpoint (step 600, macro BA 55.59%) initializes the current short paired continuation. See [RESULTS.md](RESULTS.md) for the completed comparison and preprocessing probe evidence.

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

## Current matched preprocessing experiment

The current run is `cross_subject_fold0_leaf_preprocessing`. Two one-H200 branches, `baseline` and `leaf_raw`, restore the identical version-2 `cross_subject_fold0_bn_control/frozen/best.pt` checkpoint at global step 600, including optimizer, scheduler and RNG states. Both freeze BN statistics while leaving LEAF parameters and JEV LoRA trainable. The baseline retains the online adapter for all tasks; `leaf_raw` overrides only BNCI2014001 with the verified raw LEAF cache. Other tasks retain identical inputs and grouping order. The full 11-dataset sampler, all labels and splits, batch 88, microbatch 22, eval microbatch 32 and learning rates are unchanged.

Run 200 additional optimizer updates, 601–800. Preserve the original 3000-step learning-rate horizon and the existing encoder warmup history; do not reset optimizer moments or reinitialize a projection/head. Validate the same 9,748 rows before any update (step 600) and at 700 and 800. The first validation establishes whether the input replacement changes predictions before adaptation. Within each new branch, reset only the best-score bookkeeping, select by the existing dataset-macro BA rule and retain the initial/best/final state. The comparison of interest is the 2a accuracy change between matched branches; also report every other dataset to expose interference. This is a short diagnostic, not a claim of convergence or an independent test result.

The completed BN comparison (1200 steps) selected frozen BN; its full metrics/checkpoints remain as the source and reference. Its source state is archived in Git at `be96b7c` before this paired continuation. Do not overwrite historical outputs or delete ignored data/checkpoints.

## Complete state, validation and monitoring

Version-2 checkpoints retain all trainable named parameters, all model buffers (including BN mean/variance/count), optimizer/scheduler state and Python/NumPy/torch RNG states. Both branches save initial states and verify matching parameters/buffers/source fingerprints. Reload tests destroy the head/BN state, restore it, and require identical logits on 38 fixed examples covering every task/class. The prepared-cache manifest and hashes are saved in the new checkpoint configuration. Input preprocessing intentionally differs for 2a, so initial prediction equality across branches is not required; equality of their initial model/optimizer/RNG states is required.

Validation reports normal/reversed option accuracy, BA, confusion matrices, per-subject scores and semantic consistency. Save initial, last (every 100 updates, plus first update), and validation-selected best weights. No test-set access or automatic extension beyond step 800. `monitor.json` / `LIVE.md` summarize training and validation and flag collapsed classes or >=10-point BA declines. These are diagnostic flags, not proof of causes.

The launcher starts an independent TensorBoard exporter alongside training. `tensorboard/baseline` and `tensorboard/leaf_raw` record per-step overall and per-dataset training accuracy/loss, learning rates and gradient norms. Validation records per-dataset accuracy/BA/loss and overall dataset-macro accuracy/BA/loss; `*_pooled` metrics weight examples instead. All accuracy values are fractions. Validation loss is cross entropy reconstructed from the true semantic-class probability. Both option orders are recorded. JSON/JSONL remains authoritative, and export can be resumed without duplicating already indexed rows after a clean stop.

## Entry points and files

Run `laya/.venv-eeg/bin/python scripts/run_benchmark.py`. It runs matched two-update smoke tests first, then restores the shared step-600 checkpoint for the two real branches. Source hashes guard the active code, configs and TensorBoard exporter. Existing output directories are rejected. SIGTERM requests a checkpoint at the next update boundary.

- Run root: `artifacts/benchmark/cross_subject_fold0_leaf_preprocessing/`.
- Branch logs and weights: `{baseline,leaf_raw}/metrics.jsonl`, `initial.pt`, `last.pt`, `best.pt`.
- TensorBoard: `<run root>/tensorboard/`; open with `laya/.venv-eeg/bin/python -m tensorboard.main --logdir <run root>/tensorboard --port 6006`.
- Shared original signals/splits: `datasets/benchmark/signals/`, fold 0.
- 2a override: `datasets/benchmark/leaf_2a_preprocessing/`; only its declared training/validation rows are supported.
- Historical paired BN metrics and frozen probes: `artifacts/benchmark/cross_subject_fold0_bn_control/`.

## Supporting frozen-representation diagnostics

The completed source probes below used the pre-continuation code at `be96b7c`; use an isolated worktree at that commit to reproduce checkpoint extraction with its strict source checks. Existing feature-cache CV remains usable without re-extracting.

While the matched branches run, `scripts/diagnose_representation.py` freezes the saved frozen-BN step-200 LEAF encoder, including its BN buffers, and extracts the exact eight query tokens used as JEV input. Flattening produces 2,048 features per example. Three unresolved datasets are checked: BNCI2014001, BNCI2014009 and Nakanishi2015. Use all their existing training rows and the same held-out validation subjects; no test access or split changes.

Each task gets a separate fixed `StandardScaler` + `LogisticRegression(C=1, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)` pipeline. Scaling and classifier fitting use training rows only. No validation hyperparameter search. Save train/validation accuracy, BA, confusion matrices, convergence warnings, IDs, probabilities, features and the exact encoder snapshot under `cross_subject_fold0_bn_control/representation_probe_0200/`. Verify encoder parameters and buffers remain unchanged after extraction.

This is a diagnostic of accessible class information, not a matched comparison of optimizers or training exposure. A better linear-probe result would support usable information in the encoder output, but would not by itself locate a specific JEV interface bug. The extra encoder-only inference process may affect one branch's wall time; it does not update either training model.

The fixed frequency positive control can also be applied to validation with `scripts/audit_benchmark.py --frequency-check-validation [--online-input] --out <path>`. It uses the same candidate fundamentals, Hann window and per-channel energy normalization fixed for the training-only check. This is a validation diagnostic, not test evaluation or hyperparameter selection.

To check whether the fixed-C probe's generalization gap is explained by weak regularization, the same helper accepts `--source-cv-cache <representation_probe_0200> --out <representation_probe_0200/source_cv>`. On CPU, it selects C from `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]` using leave-one-training-subject-out BA, averaged equally across subjects. Refit the scaler and classifier inside each fold; break ties toward stronger regularization. Save all selections before reading outer validation arrays, then refit each selected model on all source rows and evaluate the fixed validation set once. No test data or JEV updates. The encoder was already trained using all source subjects, so inner-fold scores diagnose the readout and are not unbiased end-to-end unseen-subject estimates. This follow-up also does not turn the repeatedly inspected outer validation set into independent test evidence.

A matched initial-encoder control uses `frozen/initial.pt` (step 100, before any encoder update) with the identical extraction, fixed-C and source-subject-CV procedures. Its encoder matches the release weights and original BN statistics. Store it under `representation_probe_0200/initial_encoder_control/`; this is a diagnostic control within the current experiment, not a new training approach. It separates limitations already present in the initial query representation from changes after joint fine-tuning.

The bounded 2a instruction control uses `--native-candidate-instruction --datasets BNCI2014001 --expected-step 100` with the same initial checkpoint. Following upstream `b_instruct_tuning.py` / `c_inference.py`, mix the task vector with the normalized mean of the four cached native candidate vectors (`Left`, `Right`, `Foot`, `Tongue`), then normalize again. Every example receives the complete candidate set; its true class is never used for conditioning. Keep signal preprocessing, all encoder weights, BN statistics, train/validation IDs and probe recipes fixed. Compare with the already extracted task-only initial queries using fixed-C probes, the same source-subject C selection and the released output projection/prototype scores. Save under `representation_probe_0200/native_candidate_control/`. This checks a supported upstream input variant; it does not assume task-only conditioning is a bug or change either live training branch.

## Raw LEAF 2a preprocessing comparison

Use the same 2,016 training and 288 validation T-session trials as the completed BN comparison. The baseline is the existing initial-encoder query/probe control. The intervention reloads original `A02T.mat`–`A09T.mat` under `/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/MNE-bnci-data/~bci/database/001-2014/`, applies the LEAF 2a raw pipeline, and retains the same IDs, complete label sequences, subjects, class texts, encoder weights and readout procedures. Do not read the held-out A01 signal or add E-session trials. This is a bundled preprocessing comparison, not an isolated CAR effect.

Processing: select labeled runs; take the same six seconds with upstream sample indexing; CAR on the 22 original channels; MNE FIR 0.3–40 Hz; resample to 200 Hz; keep seconds 2–6; call upstream `shared.pipeline` for percentile clipping, median/IQR scaling and 65-channel interpolation. The upstream hard-coded skip of the first three runs drops 96 valid A04T trials, so run selection instead checks for trial labels. Verify exact upstream raw epochs on the common runs. Raw label sequences must match every current subject's cached trial order before export.

LEAF robust scaling pools all unlabeled T-session epochs of each subject separately. This includes validation-subject statistics and is a transductive/batch preprocessing convention, not single-trial deployment; no labels enter normalization. Because only T trials are retained, its normalization pool differs from upstream's combined E+T pool. Raw preprocessing also has the upstream six-second filter context that the four-second benchmark cache no longer contains. Keep these limitations explicit when interpreting a change.

Prepare with `scripts/prepare_benchmark.py --leaf-2a-raw-root <raw directory> --out datasets/benchmark/leaf_2a_preprocessing`. Then use `scripts/diagnose_representation.py --checkpoint <run>/frozen/initial.pt --expected-step 100 --datasets BNCI2014001 --prepared-signals datasets/benchmark/leaf_2a_preprocessing --out <run>/preprocessing_control_2a/leaf_raw_probe`. Freeze all encoder parameters and BN statistics; keep task-only conditioning. Compare fixed-C=1 probes, the existing source-subject C grid/selection, and released output-projection readouts. The already inspected outer validation set remains diagnostic, not independent test evidence; released LEAF pretraining exposure is not excluded. Save this control within the current run and retain the original inputs/checkpoints.

## Known-answer option-binding diagnostic

`scripts/diagnose_option_binding.py` checks all 38 task/classes with cyclic option rotations (222 cases per condition). Select exactly one deterministic **training** example per class. In the oracle conditions, the true category is deliberately stated in text, without revealing its letter. These scores measure following known categories and option positions, never EEG accuracy or generalization.

First evaluate the native JEV adapter/head on its documented bare text template and on the EEG wrapper with no signal tokens. Then restore the frozen-BN step-400 model and repeat those two controls, plus actual EEG without a hint and actual EEG with the known-category hint. The signal-only control also has only 38 distinct source examples and is not a full-training-set metric. All six conditions use the same label-independent cyclic orders, covering every answer position for every class. Positive shared temperature calibration is omitted because it cannot change argmax.

This diagnostic uses a separate inference process, no optimizer, and does not mutate either live model. Require at least 65 GiB of spare GPU memory before loading it. Save prompts, probabilities, the restored model state, and state-immutability checks under `cross_subject_fold0_bn_control/option_binding_0400/`. Its GPU work may affect a branch's wall time but does not alter that branch's training semantics.

Superseded tracked implementations are in Git (`521f865` before this replacement). Old artifacts are moved outside the repository, not deleted. See [ARCHIVE.md](ARCHIVE.md) for the inventory. Unrelated ECG-R1 working files, shared models, datasets, environments and third-party installations are retained.
