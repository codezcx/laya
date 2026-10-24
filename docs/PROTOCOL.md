# Task-conditioned frozen LEAF → JEV on EEG-FM-Benchmark

Current implementation extends the LEAF → JEV approach to the benchmark classification tasks. Catalog: `configs/benchmark_tasks.json`; run configuration: `configs/benchmark_train.json`. Previous 2a/SEED protocol is archived in Git at `0391a40`; its caches, checkpoints and logs are retained. The old two-task launcher remains a matched control using the same trainer.

## Scope and splits

Local upstream: `../EEG-FM-Benchmark`, commit `cc196f64ac4c55627e988cf3e7d55b58aa363a41`. Eleven classification datasets: BNCI2014001 (4), BNCI2014004 (2), BNCI2015001 (2), BNCI2014008 (2), BNCI2014009 (2), CHB_MIT (2), TUAB (2), SleepEDF (5), SEED (3), EEGMAT (2), Nakanishi2015 (12). Parenthesized values are numbers of options. SEED_VIG regression and ThingsEEG2 image-feature matching remain pending: raw data not located and output protocol awaiting clarification. This is NOT a completed 13-dataset reproduction.

One shared model; fixed approximately 70/15/15 subject-disjoint train/validation/test partition, seed42. Small datasets use rounded subject counts. Explicit mappings/counts are in each dataset's `complete.json`; rows carry original indices and split. chb01/chb21 are grouped as one person, per [PhysioNet](https://physionet.org/content/chbmit/1.0.0/). Existing cache usage flags encode different few-shot splits and are not used for this new holdout protocol. TUAB cache contains original train+eval; this protocol repartitions their subjects and is NOT evaluation on TUAB's original official test split. Do not compare directly to official LOSO/10-fold/few-shot tables.

All cached epochs remain available; no training dataset is capped. Sampling balances dataset, class, then subject; global batch88 gives8examples/dataset/step. 3,000steps is an initial SFT run, NOT one traversal of every epoch: sampling uses replacement and large datasets such as sleep will not be exhausted. Validation monitors a fixed, class-stratified subset capped at2,048epochs/dataset every200steps; after selecting the best checkpoint, separately evaluate FULL validation and FULL held-out test sets. Selection uses the unweighted mean of dataset balanced accuracies. Test is never used for checkpoint selection.

## Input and text

Use the trusted local benchmark pickles without redefining labels. BNCI2015001 is right hand/feet; both P300 datasets map0=target,1=non-target. Sleep labels are wake,N1,N2,N3,REM. Nakanishi mounted MOABB epochs retain12numeric frequencies in increasing order. SEED here has1-second cached epochs, unlike the older4-second matched control.

Resample to200Hz; anti-alias filtering occurs in polyphase resampling. Preserve the source preprocessing; do not apply MI-specific bands to all tasks. Robust0.1/99.9percentile clipping and median/IQR scaling are computed separately per complete epoch, preventing cross-sample statistics. This normalization scope differs from upstream LEAF's batch-wide pipeline. Map to the released LEAF65-channel template using its aliases/interpolation; SEED additionally clips[-10,10]. A1/A2 reference electrodes in TUAB are excluded because they are absent from the LEAF template; retain all19scalp channels. Sparse montages entail extensive interpolation and may limit performance.

CHB-MIT and SleepEDF are bipolar. Use an incidence-matrix pseudoinverse to obtain minimum-norm electrode representations; each connected component has an arbitrary zero-mean reference. This does NOT recover physical monopolar EEG. Missing CHB leads filled with zeros are excluded from the linear constraints. Require reconstruction residual below1% on observed leads; fail rather than silently train on inconsistent montages. All-zero/nonfinite epochs are excluded with original indices and reasons in each cache manifest; constant epochs with undefined IQR or inconsistent observed bipolar montages still fail preparation and require inspection.

Keep full epochs: SleepEDF30seconds becomes3ordered10-second segments; concatenate their8query tokens into24tokens. Other inputs use8tokens. Zero-pad incomplete100-sample tokenizer patches; no tail is discarded. JEV masks padded query slots. Temporal order is retained through sequence positions.

LEAF is frozen in eval mode, loading the original released checkpoint strictly. The task instruction alone conditions its QFormer; no label enters the encoder. New task instructions use original `sentence-transformers/all-mpnet-base-v2` mean pooling plus L2 normalization; existing instruction vectors match the released cache with cosine>0.9999. Text cache and checkpoint remain ignored artifacts. Each segment exposes8×256 final-normalized query features.

JEV starts from native `laya/JEV-27B-VL`, NOT the prior2a/SEED adapter. Train256→5120connector, native JEV LoRA and16-position choice head; freeze the language base. Question and options explicitly name the task and candidate semantics. Training shuffles options with label-independent deterministic permutations, remaps targets, and masks unavailable output positions. Loss is answer-position cross entropy. Inference reports normal/reversed-option accuracy, BA, confusion matrices, per-subject metrics and semantic consistency. These benchmark caches do not contain parent-trial identifiers; report epoch-level metrics, NOT SEED parent-trial accuracy.

## Reproduction and artifacts

1. `python scripts/prepare_benchmark_text.py` (MPNet local files in `artifacts/benchmark/mpnet`).
2. `python scripts/audit_benchmark.py` checks real signals, labels, units, channels and segment shapes.
3. `python scripts/run_benchmark.py` prepares on4GPUs, runs1-step distributed smoke training/evaluation, then SFT and full evaluation automatically. Use `laya/.venv-eeg/bin/python` for these commands.

Cached data: ignored `datasets/benchmark/<dataset>/`. Progress: `artifacts/benchmark/pipeline.json`; training: `artifacts/benchmark/jev/metrics.jsonl`; checkpoints: `jev/best.pt`, `jev/last.pt`. Final evaluation: `final_validation/complete.json`, `final_test/complete.json`. Failures are written to pipeline state; do not interpret cached features or smoke accuracy as final SFT results.
