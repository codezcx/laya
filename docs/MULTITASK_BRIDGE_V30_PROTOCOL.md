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

## Full source token cache

`eeg_align/cache_multitask_v30.py --prepare` combines all 37,432 retained v12 MI rows and 32,827 v22 emotion rows (70,259 total). It rejects any non-source split and any within-dataset training/validation subject or recording overlap. The manifest and source keep indices are hashed. Four `--shard N` workers were launched on separate GPUs; jobs and logs are in `artifacts/multitask_bridge_v30/source_cache`. This is encoder extraction, not Qwen training.

Per-shard `.npy` memmaps store final REVE tokens as fp16 (encoder runs fp32), zero-padded to 120 tokens with explicit lengths and original global row indices. Actual MI inputs use common scaling and 12 tokens; emotion uses per-channel scaling and 120 tokens. FACED uses the already audited normalized signal caches, checks their signal hashes and label/record identity before selecting the common 30 channels. SEED is reconstructed from MAT and checked against its label file. MI is reread from pickle and its original normalization hash checked before applying common scaling. Raw recordings are kept on one worker to avoid repeated SEED filtering; shards are balanced by total token count.

After all four workers terminate successfully, `--verify` must check file hashes, exact global-index coverage, expected lengths, finite arrays and zero padding before any trainer consumes the cache. A running worker or an allocated `.npy` file is not proof of completion. The full trainer and matched baseline remain pending.

## Frozen-Qwen source training protocol

The full trainer is now implemented in `eeg_align/train_multitask_v30.py`. Its first run is fixed at 500 optimizer updates, seed 42, four GPUs, 12 samples/rank/update (three microbatches of four), AdamW lr 1e-4 and weight decay .01, norm clipping 1. Each rank samples one training example from each of the twelve dataset/class groups per update. Both tasks have six such groups, giving equal task mass. Only the connector trains. The frozen five-row vocabulary readout is not calibrated on external data.

All source validation samples are evaluated at initialization and every 100 steps, without distributed padding duplicates. Primary selection is the mean over tasks, then source datasets, then subjects, of balanced **unmasked five-class** accuracy. Diagnostic task routing accuracy and oracle-task class accuracy are separate; oracle scores cannot select the checkpoint. Step zero is eligible so degradation cannot be hidden. Tests explicitly verify that perfect oracle classification with wrong routing scores zero, and that the two tasks contribute equally despite different numbers of source datasets.

`scripts/run_bridge_v30.py` watches the already launched cache process IDs. It fails if an incomplete worker is missing/terminal and never restarts it automatically. Once all shard completion records exist, it runs full verification and launches exactly one four-GPU training job. Status is recorded in `artifacts/multitask_bridge_v30/controller.json`; the training output is `qwen_frozen/` and log is `training.log`. The matched signal-only control and external evaluation remain pending; completing this job alone does not establish language benefit or zero-shot success.

## Matched signal-only control (implemented, not yet run)

`--arm signal` in the same trainer selects `eeg_align/signal_control_v30.py`. This model applies shared LayerNorm to 512-dimensional tokens, zeroes padding, retains all 120 ordered positions, and uses a 61,440→307→5 GELU MLP. Its 18,864,951 trainable parameters are within 0.1% of the 18,883,584-parameter Qwen connector. The architectural inductive biases still differ; this is approximate parameter matching, not identical models or matched compute. It does not load Qwen or use text prototypes. Both arms use exactly the same cache, rank sampling seeds, updates, optimizer settings, validation rows and selection metric. Changing token count may expose montage/task identity in both arms.

The ordinary readout remains under the `connector` attribute so the identical optimizer/checkpoint code applies. Invalid NaN padding is removed before normalization and then masked again, avoiding contamination by LayerNorm bias. Three tests pass: invalid-padding logit/gradient safety and two selection-metric tests. The control is not automatically launched by the Qwen controller; schedule it once GPU resources are free and preserve a separate output directory.

Cache extraction finished with shard counts 17,626 / 17,743 / 17,445 / 17,445. Verification and training status must be read from their live artifacts; extraction completion alone is not yet the verification result.

## Verified cache and interim training evidence

Full verification passed for all 70,259 rows: exhaustive unique index coverage, finite values, expected token lengths, zero padding, and matching file/manifest hashes. Training uses 57,749 rows; source validation contains 12,510 rows. The four-GPU frozen-Qwen run started successfully and reached optimizer updates without DDP errors.

An additional comparison against the earlier 12-sample independent fp32 extraction initially failed a naive absolute tolerance of .02. This was investigated rather than relaxed: MI differences are exactly fp16 rounding (up to .0312); emotion differences also reflect singleton versus batch encoder computation. `scripts/audit_bridge_replay_v30.py` reconstructs the same signals and the same batch slices used during caching. All twelve input hashes match the independent smoke, and all twelve matched-batch fp16 token arrays replay exactly (maximum difference zero). Evidence is retained in `source_cache/smoke_replay_audit.json` and `source_cache/matched_batch_replay.json`. This is a targeted replay check, not an exhaustive raw-data replay of every window.

At step 200, source subject-balanced joint accuracy was Cho 58.40%, Dreyer 57.13%, Lee 49.60%, SEED 37.85%, FACED 34.97%; the task-macro selection score was 45.73%. Task routing was 100% for all sources. This can exploit the three-channel versus thirty-channel acquisition format; it is not evidence of semantic task discovery. These are interim source validation scores, not external zero-shot accuracy, and do not change the predeclared 500-step schedule.

## External paired evaluation protocol

After both arms completed and their source-selected checkpoints were frozen, `eeg_align/external_v30.py --shard N` was launched on four GPUs. It rereads all 2,840 2b E-session trials and the exact prior 6,400 SEED-VII Happy/Neutral/Sad windows. Emotion labels are checked against the original spreadsheet. All source-selected checkpoint/metadata hashes are checked before target data reading. Source and target recording paths must be disjoint. Neither target fits a normalization statistic or model parameter.

Final REVE tokens are extracted in fp32 and cached in fp16 with the same source preprocessing: three channels/common scaling for MI, common thirty channels/per-channel scaling for emotion, full-record 1–30 Hz filtering at 200 Hz, 600-sample windows, replication padding to four REVE patches. Each shard has 710 MI and 1,600 emotion windows, with hashes for arrays, lengths and ordered rows. Both models consume the identical cache in evaluation mode.

`--evaluate` checks all shards and frozen models, then saves all five logits per sample for both arms. Main MI reporting is ordinary trial accuracy; emotion reporting is subject-macro balanced accuracy. Joint five-class prediction is primary; oracle-task masking and routing are separate diagnostics. Permuting predictions within each target subject is the fixed-prompt equivalent of permuting complete EEG inputs within that subject, retaining acquisition context. It is a negative control, not a separately trained model. Paired intervals resample subjects (10,000 replicates, seed 2026); MI uses subject accuracy differences and emotion subject-balanced differences. No checkpoint or hyperparameter is chosen from these target scores.
