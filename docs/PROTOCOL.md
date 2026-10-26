# LEAF → JEV: matched task-mixture experiment

_Current protocol, fixed before launch on 2026-10-26. One seed and one subject-disjoint fold; a diagnostic, not a confirmatory benchmark._

## 🎯 Question and matched controls

Does adding SEED impair 2a performance through the shared JEV interface even when LEAF is frozen, and does updating LEAF add further interference? Reuse the current model and task catalog; do not introduce PCGrad, another encoder, or an eleven-task sweep in this stage.

| Branch | Training data | LEAF updates | GPU |
| --- | --- | --- | ---: |
| `single_frozen` | 2a | None | 0 |
| `joint_frozen` | 2a + SEED | None | 1 |
| `single_finetuned` | 2a | Steps 101–600 | 2 |
| `joint_finetuned` | 2a + SEED | Steps 101–600 | 3 |

All four train the connector, native JEV LoRA and choice-head weights/bias. Native Qwen backbone weights remain frozen. LEAF BN running statistics and dropout are fixed in all arms: LEAF stays in eval mode, with autograd enabled only for fine-tuned arms after warmup. Eval mode does not disable parameter gradients. Unused reconstruction heads, mask token and text-output projection remain outside optimization.

Initialize all arms from released LEAF and native JEV, with the same randomly initialized connector (seed 42). Do not resume the previously mixed-trained step-600 checkpoint. GPUs are identical H100 80GB devices, one process each. Hardware assignment is fixed; this is not an independently replicated or randomized device study.

## 📋 Data and input access

Use unchanged `datasets/benchmark/signals/fold0` manifests. Both datasets hold subject 1 for test, subject 2 for validation, and subjects 3 onward for training. The 2a task has 2,016 training / 288 validation trials; SEED has 44,122 training windows and a fixed class-balanced 2,046-window subset of its 3,394 validation windows. Validation rows are identical across all arms. No test evaluation or test-label selection is scheduled. Single-task arms also report SEED validation as an **untrained-task diagnostic**, not trained SEED performance.

All arms use the same verified `datasets/benchmark/leaf_2a_preprocessing` override for 2a: T-session labeled runs, upstream six-second extraction context, CAR on 22 channels, FIR 0.3–40 Hz, resampling to 200 Hz, final four-second activity segment, robust scaling and interpolation to 65 channels. Cache IDs and labels match the original fold. Only train/validation rows exist in this cache. SEED retains per-epoch clipping, median/IQR scaling, 200 Hz and 65-channel mapping, followed by clipping to ±10.

Important limitation: 2a normalization pools each subject's unlabeled T-session epochs, including the validation subject's batch. This is transductive/batch preprocessing, **not strict single-trial zero-shot deployment**. No class labels enter normalization. Released LEAF pretraining overlap is not excluded. The outer validation subject has been repeatedly inspected; it is a development endpoint, not an untouched test result.

Only relevant signal stores are initialized; montage geometry is constructed from channel metadata, not sample waveforms. Labels are used for sampling and loss, not LEAF instructions. The eleven-task catalog remains stable so cached instruction indices and class semantics are unchanged.

## ⚙️ Model, objective and budget

Task-only text conditions LEAF's QFormer. Each 2a/SEED example produces eight 256-dimensional query tokens. The connector maps them to 5,120 dimensions; JEV receives signal embeddings, task question and randomly permuted options ending at `[decision]:`. Use its native 16-position choice head with invalid positions masked. No image encoder, auxiliary subject branch, temperature fitting or generated chain of thought is added.

Each update contains exactly eight 2a examples, balanced across four classes and sampled by subject. Joint arms add eight SEED examples. Per-task RNG is keyed by seed, update and stable task index, so adding SEED cannot change the 2a sample stream. Option permutations are keyed by seed, update and sample ID, independently of the true class. Train/evaluation microbatch size is eight.

```text
single objective = mean(CE_2a)
joint objective  = mean(CE_2a) + mean(CE_SEED)
```

Do not average joint loss over sixteen examples: that halves the 2a coefficient. Sum gradients from both task microbatches before one optimizer step. All arms receive 600 updates and 4,800 2a sample exposures; joint arms add 4,800 SEED exposures. This matches anchor exposure, not total examples or FLOPs. Tasks still interact through summed gradients, global clipping and Adam moments; equal loss coefficients do not guarantee equal effective update contributions. `objective_loss` is the sum of task means; ordinary `loss` is mean CE per sampled example.

| Setting | Value |
| --- | ---: |
| Optimizer / weight decay | AdamW / 0.01 |
| Connector peak LR | 1e-4 |
| JEV LoRA peak LR | 5e-6 |
| Choice-head peak LR | 1e-5 |
| Active LEAF peak LR | 1e-6 |
| Global gradient-norm clip | 1 |
| Budget / LR horizon | 600 / 600 |
| LR warmup / final ratio | 18 active updates / 0.02 |
| Fine-tuned encoder warmup | Frozen through update 100 |

Frozen arms have zero encoder LR and no encoder gradient throughout. Fine-tuned arms receive 500 encoder updates; encoder LR warms up over updates 101–118. No automatic early stop, hyperparameter search or extension.

## 📊 Endpoints and interpretation

Primary comparison: **2a validation BA at fixed final step 600**. Also retain the best checkpoint selected by 2a BA at steps 0, 100, 200, 300, 400, 500 and 600, with earliest-step tie breaking. Never select joint arms by a two-task average. Save normal/reversed-option accuracy and BA, confusion matrices, class recall, semantic option consistency and per-subject metrics for both datasets. Training accuracy is a sampled-batch statistic, not full-training-set accuracy.

Define `Δ_frozen = joint_frozen − single_frozen` and `Δ_finetuned = joint_finetuned − single_finetuned` on the same 2a endpoint. Interaction is `Δ_finetuned − Δ_frozen`. A negative frozen contrast motivates checking shared connector/LoRA/head optimization. A more negative fine-tuned contrast motivates independently probing the encoder; it is not proof of lost representation information. If single-task performance is already poor, prioritize input/representation diagnosis.

One seed, one validation subject and repeated checkpoints are not independent replicates. Report descriptive effect sizes, not significance, convergence or general cross-dataset gains. Conditional follow-ups are frozen-feature probes, block-wise gradient conflict diagnostics, then separately tested targeted fixes. They are not automatically launched.

## 🔧 Launch, verification and recovery

```sh
laya/.venv-eeg/bin/python scripts/run_benchmark.py
```

First run four real-model two-update smoke tests concurrently, including frozen/unfrozen encoder paths. Then start four fresh formal workers. Each stage blocks at a shared gate until initial trainable parameters/buffers and the complete anchor-sampling schedule match. Checkpoint-reload tests require identical logits on seven fixed examples covering all 2a/SEED classes. During training, monitor actual anchor batch hashes across arms. Every frozen-arm checkpoint verifies all encoder parameters remain unchanged; all arms verify unchanged BN buffers.

Version-2 checkpoints contain trainable parameters, complete buffers, optimizer/scheduler and Python/NumPy/torch RNG states. Save `initial.pt`, `last.pt` and 2a-selected `best.pt`. Reload checks run at validation checkpoints. A source-hash guard stops the controller if implementation/configuration/exporter files change. SIGTERM requests saving at the next update boundary. Existing outputs are never overwritten.

Run root: `artifacts/benchmark/cross_subject_fold0_task_mixture/`. Read `pipeline.json` for process/stage status, `LIVE.md` / `monitor.json` for metrics and matched contrasts, and `<branch>/metrics.jsonl` for authoritative events. TensorBoard exports to `<run root>/tensorboard/`. Branches store configuration/source hashes, `sampling.json`, initial-state proof, validation predictions and reload checks.

## 📚 Provenance

Previous preprocessing implementation/protocol/results are committed at `5200248`. Use an isolated worktree for historical reruns. Historical checkpoints/logs under `cross_subject_fold0_leaf_preprocessing` and `cross_subject_fold0_bn_control` remain untouched; no ignored data/model files are removed. See [results](RESULTS.md) and [archive](ARCHIVE.md).

Experimental-design and resource-checking procedures assisted this protocol; they do not establish scientific validity or replace replication.[^1]

[^1]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065
