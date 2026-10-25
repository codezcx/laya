# Online EEG → text-conditioned LEAF → JEV: joint cross-subject fold

Current implementation trains one shared model on 11 classification datasets from EEG-FM-Benchmark. The current experiment adds learning-rate warmup and cosine decay to the stopped constant-LR run, archived at `d11b0bf`. Task catalog: `configs/benchmark_tasks.json`; training settings: `configs/benchmark_train.json`. Ignored datasets, checkpoints and logs are retained. No frozen-query cache is used by this run.

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

Original released LEAF checkpoint plus native `laya/JEV-27B-VL`, not previous2a/SEED adapters. Task text alone conditions LEAF's FiLM/QFormer; target labels never enter the signal encoder. Task instruction vectors use released MPNet's mean-pooling/L2 recipe, verified against existing vectors (cosine>0.9999).

Each segment emits8final-normalized256-dimensional query tokens. Sleep has24tokens; variable query padding is masked in JEV. Project256→5120, concatenate with text embeddings, and use the native16-position choice head. Text specifies task and candidate semantics; label-independent option permutations are regenerated per training step. Mask unused choices and train with answer-position cross entropy.

Steps 1–100: LEAF eval mode and no gradient, train connector/JEV LoRA/choice head. Step 101 onward: enable gradients and training mode in LEAF's tokenizer, both Transformer branches and FiLM/QFormer. This frozen-encoder interval is distinct from learning-rate warmup. Unused reconstruction heads, mask token and final text-embedding output head remain frozen because they do not lie on the query-to-JEV path. Native language-base weights remain frozen; JEV LoRA is always active. A two-step smoke run uses a one-step encoder freeze and one-step LR warmup, and explicitly checks no encoder change while frozen, then nonzero encoder gradient and parameter change after unfreezing.

### Learning-rate experiment

Fresh initialization from the same released LEAF and native JEV weights and seed 42; do not resume the old step-600/1000 model. Retain the old optimizer, weight decay, clipping, loss, dataset catalog, input processing, training/validation splits, balanced sampler and option permutations. Peak LRs remain connector `1e-4`, JEV LoRA `5e-6`, choice head `1e-5`, LEAF `1e-6`.

`eeg_align/lr_schedule.py` sets rates BEFORE each numbered optimizer update. Connector/LoRA/head rise linearly over updates 1–90 (3% of 3,000), then decay with a cosine to 2% of their peak at update 3,000. LEAF LR is zero through update 100, warms up independently over updates 101–190, then decays to 2% at update 3,000. Thus final LRs are `2e-6`, `1e-7`, `2e-7`, `2e-8`. Each metric row records the actual four rates used for that update; checkpoints include scheduler settings, last update and optimizer state. A `constant` scheduler mode exists as a matched control in the same implementation; it is not a separately maintained training script. The current launcher starts only the configured cosine run.

Compare fixed validation subsets at the same update numbers against the retained constant-LR run `artifacts/benchmark/cross_subject_fold0/`. The old run stopped at logged update 1,087, last saved update 1,000, best update 600; it provides no later-step control. World size changes from 4 to 2 and microbatch from 4 to 22 on the new node, but global batch and sample IDs/options at each step remain the same. Dropout RNG and floating-point reduction order may differ, so this is a useful first schedule comparison, not a bitwise-controlled causal proof or a multi-seed result. Warmup and cosine are changed together; this does not separately identify their effects.

Global batch 88, dataset/class/subject-balanced sampling with replacement: 8 examples from each dataset each step, split over 2 GPUs. Microbatch 22/rank with two accumulation microbatches per update; eval 32/rank. 3,000 steps is an initial SFT budget, not a full pass over every large dataset. Logs include per-dataset batch losses/accuracy, encoder gradient norm and all group learning rates. Validation uses the same fixed class-stratified subset capped at 2,048 epochs/dataset (9,748 total) every 200 steps; choose best by unweighted dataset mean BA. Save last weights every 100 steps. Final best-checkpoint evaluation uses FULL validation and FULL test. Report epoch accuracy/BA, per-subject metrics, confusion matrices and semantic consistency under option reversal. SEED parent-trial accuracy is unavailable from these caches and is not invented.

## Entry points and outputs

Run `laya/.venv-eeg/bin/python scripts/run_benchmark.py`: verify/reuse existing raw exports → split audit → configured two-GPU two-step smoke → fresh production SFT → full validation/test on best checkpoint. Source hashes prevent changes between stages. `--resume-preparation` reuses completed exports and appends logs; it does not resume optimizer training. Existing training outputs are never silently overwritten. `run_name` selects the isolated ignored artifact directory, and GPU count comes from `world_size` rather than hard-coded four-GPU launch arguments.

- Data: `datasets/benchmark/signals/<dataset>/signals.npy`, `rows.jsonl`, `complete.json`.
- Split: `datasets/benchmark/signals/fold0/{train,validation,test}.jsonl`, `complete.json`.
- Status: `artifacts/benchmark/cross_subject_fold0_lr_cosine/pipeline.json`.
- Metrics: `artifacts/benchmark/cross_subject_fold0_lr_cosine/jev/metrics.jsonl`.
- Best/last weights: same `jev/` folder (include trained LEAF parameters).
- Full eval: `final_validation/complete.json`, `final_test/complete.json`.
