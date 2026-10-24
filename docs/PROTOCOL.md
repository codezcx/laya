# Frozen LEAF → JEV: 2a four-class and SEED three-class

## Order and purpose

First extract frozen instruction-tuned LEAF outputs and evaluate native text-prototype matching. Then train JEV on the identical cached query features/splits. This is an interface adaptation experiment, not unseen-dataset zero-shot: both datasets appear in the upstream catalog and released-weight subject/record exposure is not excluded. Original REVE experiments are archived at c75a101 and prior commits; ignored data, weights and logs remain untouched. Prior MI+MAT JEV finished1,600steps before replacement.

## Data and preprocessing

2a: all four labels from original18MAT files, train subjectsA01–A06, validationA07, untouched testA08–A09. Only14train/val files are read. All labeled runs, including subjectA04's shorter run list; MATLAB trial indices converted from1-based to0-based. Average reference,6-second onset windows, upstream0.3–40Hz filtering and200Hz resampling, then[2,6)seconds. The upstream FIR on6-second epochs warns that filter length exceeds the epoch; retained for this baseline/adapter comparison and documented, not silently altered. Original label1/2/3/4→left/right/feet/tongue. Each subject576trials; train3,456,validation576. Official upstream loader instead random-splits trials of subjects1–7; this experiment is subject-disjoint for our adapter.

SEED: existing9train/3validation/3test subject mapping from datasets/mi_emotion/complete.json, all3sessions. Official preprocessed62-channel,200Hz microvolt signals converted to volts for MNE;0.1–70Hz plus50Hz notch. Process each recording/trial separately (no joining sessions or allowing windows across boundaries).4-second nonoverlap windows; all22,734train windows,8evenly-spaced windows per validation parent (1,080windows/135trials). Test subjects untouched. Labels from label.mat:negative0/neutral1/positive2. Upstream SEED training used within-subject trial partitions, so our held-out subjects may have appeared in released pretraining.

Use upstream percentile clipping,median/IQR scaling and65-channel mapping/interpolation. Scaling computed per2a session or SEED parent trial, not across unrelated train/validation subjects; these normalization groups differ from upstream's SEED partition-wide grouping. SEED clipped to[-10,10] after interpolation as upstream.2a22input channels map to65 with43interpolated; SEED62input channels map/merge to58template locations with7interpolated. Preserve upstream channel alias policies, including CB1→O2 and CB2→O1 merging; this is a model input convention, not an anatomical equivalence claim.

## Native baseline and cached interface

Released leaf-v1.0-instruct-mpnet-base.ckpt, strict loading, frozen eval mode. Native instructions: 'This is a motor imagery decoding task' and 'This is an emotion recognition task'. Prototypes from shipped MPNet cache:Left/Right/Foot/Tongue;Negative/Neutral/Positive.4-second input goes directly into model without10-second padding, an upstream-supported inference mode. Native embedding768dims; cosine scores over task candidates, softmax for parent aggregation. Baseline train/validation accuracy,BA,confusion matrices and window/trial counts in artifacts/leaf_2a_seed/baseline.json. No test inference.

JEV input is8×256 final-Q-Former queries AFTER final_norm, not tower tokens and not768dim final projection. Strictly identical query cache for baseline inputs and JEV. Cache hashes, checkpoint hash, upstream commit and source MAT hashes in datasets/leaf_2a_seed/complete.json. Frozen LEAF preprocessing and features do not use class labels; task instruction is supplied, so this does not test task discovery.

## JEV training

Original JEV-27B-VL language model/adapter from laya/JEV-27B-VL. New LayerNorm256→Linear256×5120→GELU→Linear5120×5120 connector; train native LoRA and native16-row choice head, base language weights frozen. Mask inactive answer positions.2a4choices,SEED3choices,order shuffled deterministically by sample/step/seed and target remapped. No additional InfoNCE.

600steps,global48=24MI+24SEED; uniformly sample class→subject→parent→window with replacement. Cross-entropy averaged over global48, task weights50/50. GPUs2,3; microbatch8,eval16; connector/LoRA/head LR1e-4/5e-6/1e-5,AdamW weight decay.01,gradient clip1,seed42. Save step1/every50/final,validateevery100/final. Report normal/reversed semantic window/trial accuracy and BA,permutation consistency; selection mean task trialBA,not pooled window accuracy.2a trial=one input,SEED trial averages8validation-window probabilities. Expected600steps is a fixed budget,not600epochs or guaranteed exhaustive sample coverage.

Commands: `CUDA_VISIBLE_DEVICES=2 laya/.venv-eeg/bin/python -u -m eeg_align.leaf_data`; inspect baseline, then `laya/.venv-eeg/bin/python scripts/run_joint_training.py` (two-rank smoke then production). No other GPU jobs stopped. See [LEAF_ENCODER.md](LEAF_ENCODER.md) for upstream provenance.
