# v30: Return to the actual EEG-to-language interface

Status: interface implemented; real-Qwen synthetic gradient check and 12-real-source training smoke passed. No full dataset training or external accuracy result yet.

## Motivation and scope

v23–v29 diagnose signal classification, not language alignment. Earlier fixed text prototypes do not establish a language benefit. This round restores EEG tokens as inputs to the Qwen decoder, while preserving all older experiments. The active objectives remain cross-dataset motor imagery and emotion recognition. Arithmetic/stress are not added.

## Implemented interface

`eeg_align/multitask_bridge.py` accepts actual cached REVE tokens shaped B,K,512 and a boolean validity mask. LayerNorm and a two-layer GELU MLP map them to Qwen's hidden width. Prefix embeddings, valid EEG embeddings and suffix embeddings form one causal decoder sequence. The final answer-position state is scored with the frozen original vocabulary rows for A–E. Five-way cross entropy supervises the connector; neither the decoder nor the readout trains in this initial version. Freezing the readout prevents a trainable classification head alone absorbing the adaptation.

One identical prompt lists negative/neutral/positive emotion and left/right motor imagery. No task ID, dataset name, or class mask enters forward. The loss still uses categorical labels: this is task-supervised interface learning, not free-text semantic pretraining or proof of unseen-task zero-shot performance.

## Verification

Run `PYTHONPATH=. laya/.venv-eeg/bin/python scripts/smoke_multitask_bridge_v30.py` once into a fresh output directory. It uses the exact CerebraGloss Qwen checkpoint in `configs/reve_qwen_mi_multi.json`, loads the language backbone and A–E vocabulary rows only, and checks finite nonzero connector gradients, no decoder gradients, and invariance to invalid EEG padding (including NaNs). Synthetic inputs provide no evidence of EEG accuracy or encoder compatibility. Output: `artifacts/multitask_bridge_v30/smoke/complete.json`.

## Dataset experiment still required

Completed interface check: output shape 2×5, synthetic loss 2.633366, connector gradient norm sum 97.62636, frozen decoder gradients absent, padding-logit maximum difference exactly 0, peak allocated GPU memory 16.183 GB. These numbers are engineering checks only, not training outcomes. The loaded vocabulary rows were [32,33,34,35,36], and 398 language tensors were loaded without the visual encoder.

Before training, build audited source-only token manifests from the existing MI and emotion sources with preserved subject splits. Do not silently substitute flattened channel features or spectral vectors for REVE tokens. Preserve token channel/time ordering and record preprocessing. MI has three channels while emotion uses thirty: task routing can exploit acquisition/montage differences, so report within-task classification separately and do not infer task understanding from routing accuracy.

Compare the real decoder bridge against a parameter-accounted signal-only readout on identical cached inputs/splits and shuffled-EEG controls. Keep a fixed source-only selection rule and model hashes before external evaluation. Do not use external 2b or SEED-VII to select a checkpoint. Both targets are already exposed exploratory sets, and REVE pretraining includes 2b. External improvement cannot be described as new blind validation or entirely unseen pretraining data.

The next action after the interface check is source token-cache integration and a bounded training smoke on real data. Full four-GPU training is not yet implemented by this module; do not reuse the old binary trainer unchanged.

## Completed real-source training smoke

`scripts/train_bridge_real_smoke_v30.py` selects the first training row per dataset/class from the audited MI keep indices and v22 emotion manifest. It uses all five source datasets: Cho2017/Dreyer2023/Lee2019_MI × two classes and SEED/FACED × three classes (12 examples total). This deterministic tiny selection is only an integration check, not a representative evaluation set.

Signals are read from original pickle/MAT/BDF files, full-trial filtered at 1–30 Hz/200 Hz and cropped to 600 samples. MI uses common scaling, emotion per-channel normalization. This differing preprocessing/montage can reveal task identity and is not evidence of semantic task inference. REVE outputs retain all four temporal patches: 12 tokens for MI and 120 for emotion, each 512-dimensional, in channel-then-time order. No temporal pooling or prefix-layer cache is mistaken for final encoder tokens.

The run completed ten AdamW connector updates (lr 1e-4, weight decay .01), each accumulating all twelve microbatches. Task/dataset/class weights give each sample 1/12. Qwen and A–E readout stayed frozen. Loss was finite, connector gradients were finite/nonzero, and decoder gradients were absent. Initial loss was 2.82645; the last logged pre-update loss was 0.69607 after nine updates. Intermediate loss was nonmonotonic. The corresponding tiny training-set accuracies were 16.67% and 58.33%; these are NOT validation or transfer results. The saved connector is after ten updates and was not scored separately. The training loop took 16.72 seconds, excluding model loading and raw extraction.

Artifacts: `artifacts/multitask_bridge_v30/real_smoke/{data.json,tokens.pt,train.jsonl,connector.pt,complete.json}`. Manifests, normalized signals, token cache and saved connector are hashed. FACED event labels for all three selected rows were additionally checked against the manifest in a separate process; this check is now inline for future runs. External 2b and SEED-VII were not read. The next pending step is scalable source caching, matched controls and four-GPU source-only training/validation; this smoke is not that experiment.
