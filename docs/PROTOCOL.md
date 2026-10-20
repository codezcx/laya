# Full-channel motor imagery → REVE → JEV

## Task and splits

One left/right hand motor-imagery label per trial, left=0=A, right=1=B. No TUAB, emotion, attribute questions, or baseline/rest comparison. All available exported EEG channels and the full exported task epoch are used. The 2a feet/tongue trials are excluded.

Nine source datasets: BNCI2014_001, Cho2017, Dreyer2023, Lee2019_MI, Weibo2014, Zhou2016, PhysionetMI, Kumar2024, Chang2025. Retain the audited deterministic per-dataset subject split (seed42, approximately20% subjects for validation): 49,193 training / 12,508 validation trials. A subject's sessions remain together. BNCI2014_004 (2b), 6,520 trials, is the cross-dataset test; it does not contribute to training, hyperparameters or checkpoint selection. No separate source test is claimed. Earlier experiments have already examined 2b, so this is a held-out dataset for the current run, not a never-touched benchmark. REVE pretraining overlap also precludes an unqualified unseen-data claim.

The source index retains the earlier 11-source audit; the current training allowlist excludes Brandl2020 (unmapped CFC/CCP electrodes) and GuttmannFlury2025_MI (unmapped CB1/CB2). Do not guess coordinates or drop EEG channels just to retain these datasets. Dreyer's two conflicting runs remain quarantined. The all-channel scan also found effectively flat FC4 (std about1e-10µV) in all80trials of Chang2025 subject14; quarantine these complete training trials without dropping a channel or changing validation/test membership. No other nonfinite/flat channels were found in the full68,301-trial preflight scan. Full audit and limitations: [MI_DATASETS.md](MI_DATASETS.md).

## Signal and variable length

Read trusted mounted200Hz float32microvolt epochs, already filtered by the exporter. No new resampling/filtering/reference transformation. Center each EEG channel over its complete trial, clip±1000µV, divide by100 (same physical-scale convention as the previous encoder use); no per-channel variance normalization or fitted cross-trial preprocessing statistics. Exclude only explicitly named EOG/trigger channels; all retained EEG channels require exact/case-normalized REVE position-bank names, unique and finite/nonflat. Unknown positions abort preparation. Auxiliary-channel handling is not a license to silently discard unmapped EEG electrodes.

REVE-base has200sample patches and180sample stride. The local HF implementation uses unfold and otherwise drops the trailing partial patch. To preserve every observed sample, repeat the last sample only at the right edge up to the next complete patch; record original length and extension length in each cache row. This synthetic edge extension (at most179samples) is not extra observed EEG. No entire trial is cropped to a common length. Number of EEG tokens is channels×temporal patches; e.g.22channels×4seconds→110tokens after edge extension,64×3seconds→256tokens,62×4seconds→310tokens,2b3×4.5seconds→15tokens.

Encode with frozen local REVE using actual electrode positions and actual per-trial patch count, batching only identical channel lists/padded lengths at this stage. Cache all final encoder tokens, no token pooling. Language batches right-pad token arrays with zeros; Boolean mask removes padded tokens before prefix/EEG/suffix assembly. Qwen receives its own correct sequence attention mask and position IDs. Decision is read at each sequence's actual last valid suffix token, never at padding. Encoding the held-out test with the frozen, fixed preprocessing/model does not fit anything to test labels.

## Model and objective

Original local JEV-27B-VL language base and original enabled LoRA; a newly initialized LayerNorm512→Linear5120→GELU→Linear5120 connector; original A/B decision rows and biases. No TUAB or earlier MI fine-tuning weights loaded. Freeze REVE and Qwen base. Train connector, JEV LoRA, and A/B head. Input prompt is the same for all datasets:

```
[kind] choice
[state] EEG recording begins:
<all valid EEG token embeddings>
EEG recording ends.
[question] Which hand is the participant imagining moving?
[options]
A) Left hand motor imagery
B) Right hand motor imagery
[decision]:
```

No dataset name, subject, true label, clinical text, or channel count is explicitly included in the text. The signal token layout can still reveal acquisition differences. One ordinary two-class cross-entropy per trial, no multi-window record aggregation and no auxiliary loss.

## Optimization and evaluation

Four GPUs; global batch up to64trials,2trials/microbatch/GPU, sample-weighted gradient accumulation and synchronized trainable gradients. Three full epochs,769steps/epoch,2,307steps total. Every training trial appears exactly once per epoch. Randomly shuffle each epoch, sort by token length only inside random pools of2,048trials to reduce padding, then shuffle batches. This is length bucketing, not fixed dataset blocks. Training uses ordinary sample-average CE; larger datasets contribute more examples, while selection weights datasets equally.

AdamW weight_decay0.01; peak learning rates connector3e-5,LoRA2e-6,head5e-6; linear warmup5% of steps, cosine decay to10% peak; gradient norm clip1. No class reweighting, no early stopping based on test results.

Initial full source validation, then every200steps and each epoch end. Validation uses exactly the same full epochs/channels each time. Report overall accuracy, BA, per-dataset and per-subject accuracy/recalls/confusion/AUROC. Select highest arithmetic mean of per-dataset validation BA (earliest on ties). After training, evaluate the selected checkpoint on 2b exactly once. Test metrics are trial-level; do not compare directly to TUAB record-level scores.

A two-step four-card smoke test includes both longest and shortest training examples, checks finite gradients and matching trainable-weight hashes after synchronization. Cache completion verifies hashes, manifest coverage, and original split identity. Each training run records source code, data hashes, configuration, and Git revision. Generated data/checkpoints/logs are ignored and preserved.

## Eight-GPU node: matched linear probe and JEV resume

GPU0–3 resume JEV from `artifacts/mi_jev/run/last.pt` into `artifacts/mi_jev/resumed/`. Migration interrupted logging at318, but the latest optimizer checkpoint is200; steps201–318 are replayed. Original metrics remain unchanged. Restore trainable weights and AdamW state, deterministic epoch/batch order, original learning-rate schedule and historical best checkpoint. Configuration and all unchanged training dependencies are hash-checked; no warmup restart.

GPU4–7 train `eeg_align.linear_probe` with `configs/mi_probe.json` on exactly the same cached REVE tokens and split manifests. No REVE gradients, new encoder, signal preprocessing or resplitting. Fixed readout averages temporal tokens per electrode, then averages electrodes separately in left/midline/right groups and concatenates three512-dimensional vectors. Grouping uses position-bank x coordinates; Z electrodes are midline; absent groups are zero. This preserves coarse laterality while supporting variable montage/duration; it is not the official fixed-montage flattened REVE classifier. Train-only feature means/stds are computed across distributed training shards. The sole trainable module is `Linear(1536,2)` (3,074 parameters).

Linear-probe optimization: AdamW lr1e-3, weight_decay1e-3, global batch512, ordinary trial CE, seed42; at most50epochs with patience10 using macro dataset source-validation BA. All training trials appear once per epoch; distributed shards do not duplicate/pad samples. Each epoch reports training loss/accuracy, full validation and per-dataset/per-subject metrics. Select source best, then evaluate2b once. Outputs: `artifacts/mi_reve_probe/metrics.jsonl`, `best.json`, `test_2b.json`. A successful probe demonstrates recoverable information under this fixed readout; a failed probe alone does not prove REVE lacks all useful MI information.

## Whole-dataset 2a transfer control

Run the same frozen REVE spatial linear probe with `--heldout-dataset BNCI2014_001`. Remove every2a trial from both source training and source validation before fitting the scaler or head; concatenate the former2a train/validation subsets into the final2,592-trial target test (all9subjects, both sessions, left/right only). Remaining8datasets retain their original subject splits:47,177train and11,932validation. 2b is not used in this control. Save exact per-run manifests and their hashes. Head/scaler start fresh; no9-source trained weights are reused. The frozen encoder token cache is reusable because it was generated without supervised adaptation.

Keep pooling, optimizer, seed, batch size,50epoch cap,patience10 and source-macro-BA selection identical to the earlier probe. No target-derived normalization, checkpoint selection or tuning. Output:`artifacts/mi_reve_probe_holdout_2a/`; final metrics:`test_2a.json`. This measures transfer of a linear readout of frozen REVE features, not full encoder fine-tuning or proof of absence from REVE pretraining. The earlier83.68%2a source-validation result is a different setting and must not be presented as cross-dataset transfer.

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. laya/.venv-eeg/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 -m eeg_align.linear_probe --out artifacts/mi_reve_probe_holdout_2a --heldout-dataset BNCI2014_001
```

## Paired channel-reduction diagnostic

`scripts/compare_mi_channels.py` fixes the source8-trained epoch30linear probe and training-only scaler, then re-encodes every2a target trial under22channels andC3/Cz/C4 only. Same per-channel centering, clipping/scaling and full temporal extent/edge extension; same electrode reference, labels, spatial pooling, normalization, head and0.5threshold. Both arms re-run REVE in eval mode, then quantize tokens throughfp16 to match training-cache precision. The3-channel arm never reads22-channel contextualized tokens. Verify22-channel predictions reproduce all original decisions with maximum probability drift below0.001; save paired predictions, per-subject results and correct/error transitions. No extra training or target calibration. Run onGPU4 while JEV continuesGPU0–3.


## Global mean matched control

Run `eeg_align.linear_probe --pooling mean --out artifacts/mi_reve_probe_mean` on GPU4–7. Reuse exactly the original nine-source split (49,193 train, 12,508 validation; 6,520 held-out 2b trials) and frozen REVE cache. Average all valid tokens into one 512-dimensional feature, without anatomical grouping or padding; retain training-only feature standardization. Train only Linear(512,2), 1,026 parameters. Same AdamW, batch512, seed42, 50-epoch cap, patience10 and macro dataset validation BA selection as the spatial probe. Evaluate 2b once after selection. This is a matched control of the current approach; JEV GPU0–3 code and training remain unchanged.

## MI JEV fitting diagnostic

`scripts/diagnose_mi_jev.py` runs two matched train-only diagnostics on GPU4–5 (MI recipe) and GPU6–7 (TUAB recipe), without editing the live JEV training dependencies. Seed42 selects four left and four right trials from each of nine training datasets (72 total). Both arms use the same raw cached tokens, original JEV initialization, fresh connector, fixed batches of16 (eight per label), frozen REVE/base, trainable connector/LoRA/head, and60steps. MI arm uses original learning rates and first60steps of its115-step warmup; TUAB arm uses constant1e-4/5e-6/1e-5. Batch16 is a diagnostic setting shared by both arms, not a reproduction of the full MI batch64 or TUAB eight-window aggregation. Every10steps evaluates all72training examples singly; these are fitting metrics, not validation/generalization scores. Logs include gradient norms per parameter group. Initial padding check compares shortest/longest selected trials individually and batched, then poisons invalid padding to verify removal. Results under `artifacts/mi_jev_diagnostic_{mi,tuab}/` with exact manifests/config/hashes. This contrasts learning-rate recipes, not individual learning-rate versus warmup effects.


## Full MI control with TUAB learning-rate recipe

Launched on GPU4–7 from isolated worktree `../eegjev-worktrees/mi-tuab-lr`, branch `experiment/mi-tuab-lr-control`, implementation commit `5b437e8`. The shared repository's original live JEV code remains hash-identical. Command from that worktree: `CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. laya/.venv-eeg/bin/python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 --master-port=29643 -m eeg_align.train_jev --recipe tuab-lr --out artifacts/mi_jev_tuab_lr`. Shared data/checkpoint paths are symlinked, with no duplicated datasets.

Fresh original JEV adapter/head and fresh seed42 connector, not diagnostic or TUAB-trained weights. Same49,193 training trials,12,508 validation trials, batch64/microbatch2, three epochs(2,307steps), validation every200steps plus epoch boundaries; source macro dataset BA selection, final2b evaluation. Only learning-rate recipe changes: connector1e-4, LoRA5e-6, head1e-5, constant from first step, no warmup or decay. Retain individual-trial CE; no TUAB record/window grouping. Output `artifacts/mi_jev_tuab_lr/` and log `artifacts/mi_jev_tuab_lr.log`. GPU0–3 retain the original lower-rate control.

Node migration: the replacement node exposes four H100 GPUs numbered0–3. Resume the TUAB-rate MI control on those four GPUs using the same isolated worktree and `--recipe tuab-lr --resume artifacts/mi_jev_tuab_lr/last.pt --out artifacts/mi_jev_tuab_lr_resumed`. All saved source hashes verified against the worktree before launch. Latest logged step348 was not checkpointed; restore model and AdamW from step200 and replay201–348 using the original deterministic ordering. Preserve the prior best checkpoint and constant rates. Original artifacts remain untouched. Launch metadata: `artifacts/mi_jev_tuab_lr_resume_launch.json`; runtime log: `artifacts/mi_jev_tuab_lr_resumed.log`. Only this requested task is resumed on the four visible GPUs.

## Fixed step1600 source-training evaluation

User requested pausing the TUAB-rate MI job to distinguish fitting failure from held-out-subject generalization failure. Verified `artifacts/mi_jev_tuab_lr_resumed/{best,last}.pt` both at1600, with optimizer in last.pt; stopped launcher at logged1632. No emergency1632checkpoint exists: future continuation must replay1601–1632. Training stays stopped pending further instruction. Run `scripts/evaluate_mi_train.py` on four visible GPUs, loading best.pt with original frozen backbone/encoder cache; no optimization. Evaluate every source training trial, prioritize2a,Physionet,Weibo,Zhou, save each dataset's predictions/accuracy/recalls immediately, then complete other datasets. Compare against the existing step1600validation, not later checkpoints. Output `artifacts/mi_jev_train_eval_1600/`; final summary `metrics.json`. Exact cached split and model dependency hashes are checked.

User subsequently requested continuation. Stopped the train-evaluation launcher after five saved datasets (2a,Physionet,Weibo,Zhou,Chang;9,801trials); retained all outputs and marked evaluation interrupted, not complete. Verified all checkpoint source hashes and restored step1600model/optimizer from `artifacts/mi_jev_tuab_lr_resumed/last.pt` in the same isolated worktree. Four-GPU training output now `artifacts/mi_jev_tuab_lr_continued/`, log `artifacts/mi_jev_tuab_lr_continued.log`, original2307-step endpoint and constant TUAB rates unchanged. No changes to data, schedule, trainable parameters, or deterministic batch ordering; replay unsaved1601–1632. Best checkpoint from1600is carried forward. Launch metadata:`artifacts/mi_jev_tuab_lr_continue_launch.json`.
