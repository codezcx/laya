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
