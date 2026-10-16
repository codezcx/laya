# v30: Return to the actual EEG-to-language interface

Status: interface implemented; real-Qwen synthetic gradient check passed. No dataset training or new accuracy result yet.

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
