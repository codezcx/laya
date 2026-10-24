# Online EEG → text-conditioned LEAF → JEV: joint cross-subject fold

Current implementation trains one shared model on11classification datasets from EEG-FM-Benchmark. Task catalog: `configs/benchmark_tasks.json`; training settings: `configs/benchmark_train.json`. Prior frozen-feature implementation is archived at `f4742a4`; ignored datasets, features, checkpoints and logs are retained. No frozen-query cache is used by this run.

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

Steps1–100: LEAF eval mode and no gradient, train connector/JEV LoRA/choice head. Step101onward: enable gradients and training mode in LEAF's tokenizer, both Transformer branches and FiLM/QFormer. Encoder LR1e-6; connector1e-4, JEV LoRA5e-6, choice head1e-5. Unused reconstruction heads, mask token and final text-embedding output head remain frozen because they do not lie on the query-to-JEV path. Native language-base weights remain frozen; JEV LoRA is always active. A two-step smoke run explicitly checks no encoder change in warmup, then nonzero encoder gradient and parameter change after unfreezing.

Global batch88, dataset/class/subject-balanced sampling with replacement:8examples from each dataset each step, split over4GPUs. Microbatch4/rank, eval8/rank. 3,000steps is an initial SFT budget, not a full pass over every large dataset. Logs include per-dataset batch losses/accuracy and encoder gradient norm. Validation uses a fixed class-stratified subset capped at2,048epochs/dataset every200steps; choose best by unweighted dataset mean BA. Final best-checkpoint evaluation uses FULL validation and FULL test. Report epoch accuracy/BA, per-subject metrics, confusion matrices and semantic consistency under option reversal. SEED parent-trial accuracy is unavailable from these caches and is not invented.

## Entry points and outputs

Run `laya/.venv-eeg/bin/python scripts/run_benchmark.py`: CPU raw export → split audit → four-GPU two-step smoke → fresh production SFT → full validation/test on best checkpoint. Source hashes prevent changes between stages. `--resume-preparation` reuses completed exports and appends logs; existing training outputs are never silently overwritten.

- Data: `datasets/benchmark/signals/<dataset>/signals.npy`, `rows.jsonl`, `complete.json`.
- Split: `datasets/benchmark/signals/fold0/{train,validation,test}.jsonl`, `complete.json`.
- Status: `artifacts/benchmark/cross_subject_fold0/pipeline.json`.
- Metrics: `artifacts/benchmark/cross_subject_fold0/jev/metrics.jsonl`.
- Best/last weights: same `jev/` folder (include trained LEAF parameters).
- Full eval: `final_validation/complete.json`, `final_test/complete.json`.
