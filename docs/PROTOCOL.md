# TUAB-only protocol

## Task and splits

One clinical normal/abnormal classification target per EDF record, normal=0=A and abnormal=1=B. No auxiliary QA or prior MI/emotion data. Source is `/inspire/dataset/tuab/v0/v3.0.1/edf`: 2,717 official training and 276 official evaluation records. A subject-stratified 15% validation subset is drawn from official training (seed42). Subjects with mixed labels stay together; strata are subject label sets. Official eval is final test, used only after model selection. Final included counts and exclusions are written to `datasets/tuab/audit.json`.

## Signals and label granularity

19 scalp electrodes: Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2. Legacy T3/T4/T5/T6 names map to T7/T8/P7/P8. Missing/duplicate required electrodes, nonfinite signals, flat selected channels or insufficient duration are excluded and reported. More than5% excluded stops the pipeline.

Original reference is retained. Read the first up to1202s, convert volts toµV, resample to200Hz, apply fourth-order zero-phase Butterworth0.5–45Hz over the continuous signal. Candidate windows lie within the first1200s; skip first60s if enough duration remains, otherwise start from0. Use disjoint equal temporal strata, selecting their midpoint10s windows. Train pool16 windows/record; validation/test8 fixed windows/record. Center each channel per window, clip±1000µV and divide by100; do not standardize each channel's variance. This is a fixed design choice, not a claim of an optimal TUAB preprocessing recipe.

During each training epoch pick one window from each adjacent pair in the16-window pool:8 windows/record, deterministic by seed,record ID,epoch. This allows limited temporal resampling, not an unlimited pool. Validation/test positions never change.

TUAB labels belong to records. No window-level hard-label CE is applied. Each window produces two logits; average eight logits, then apply one record-level CE. Average aggregation may dilute localized abnormalities; this initial experiment tests a simple fixed aggregation.

## Encoder and JEV

Frozen local REVE-base (config `configs/encoder.json`) uses19-channel positions and2000samples:11 temporal patches per channel,209tokens/window,512dimensions/token. Cache final transformer tokens without pooling. A trainable LayerNorm512→Linear5120→GELU→Linear5120 connector projects them into the language embedding dimension. Prefix+EEG tokens+fixed question/options feed the JEV language decoder. Read the last valid hidden state through original JEV A/B decision rows.

Original local `laya/JEV-27B-VL` language adapter stays enabled and is trainable. No4380-step connector, no MI/emotion trained adapter or decision rows are loaded. Connector starts randomly, seed42. Qwen base and REVE are frozen. Only connector, original JEV LoRA and two decision rows/biases train.

## Optimization

Five full epochs,16 records/global batch on4GPUs,8 windows/record,2windows/microbatch. Constant learning rates: connector1e-4,LoRA5e-6,head1e-5; AdamW weight_decay0.01,gradient clip1. Shuffle records without replacement each epoch; include final short batch with correct sample weighting. No class weighting since the source corpus is near balanced; actual split balance is reported.

Record aggregation uses a two-pass gradient cache: obtain window logits without retaining activations, differentiate record CE with respect to those logits, recompute window microbatches and backpropagate the corresponding gradients. No dropout is allowed, and recomputed logits are checked for equality. Unit test compares this gradient to ordinary whole-bag autograd. Four-card smoke checks trainable parameter hashes after each step and finite gradients.

Each epoch exposes every retained training record once; one record contributes one classification target. 8windows do not mean8independent labels. Loss and accuracy are record-level. Initial full validation, then every50steps and at epoch ends. Select highest validation record BA, keeping earliest on ties. Evaluate best checkpoint on final test once, plus validation with record identities shuffled as a signal-use control. Store prediction probabilities and IDs.

## Matched simple readout

Frozen REVE mean over tokens and eight windows→StandardScaler→logistic regression. Fit scaler on training only, select C among0.01,0.1,1,10 by source validation BA. Use same record splits and epoch0window choices for train, same fixed validation/test windows. This control has different capacity and optimization and no epoch-varying windows; it assesses whether frozen representations support the task, not a strict one-component ablation. Its test is evaluated after the JEV run finishes.

## Reproducibility and limits

Data, caches,logs,checkpoints are ignored byGit. Source code revision, data manifest hashes,token-shard hashes and encoder paths are recorded. No clinical text, directory names,subject IDs or ground-truth labels enter the language prompt. Metrics compare records rather than correlated windows. This is source-dataset supervised generalization, not cross-dataset zero-shot. REVE's pretraining overlap must be audited before making unseen-data claims. Success onTUAB alone cannot establish that MI failure was caused by insufficient data.
