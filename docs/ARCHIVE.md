# Repository archive

Historical tracked implementations can be recovered from Git. `521f865` is the state immediately before the current BN-control replacement and cleanup. The working tree keeps the current benchmark implementation and its matched controls/tests.

Historical artifacts were moved, not deleted, to:

`/inspire/hdd/global_user/zhangchenxi-253108310322/eegjev_archive/20261025_pre_bn_control/`

The archive's `index.json` maps each original repository path to its archived location and records file counts and logical byte sizes. A copy is retained at `artifacts/archive_index.json`. No compatibility symlink trees are created. To rerun old code, use an isolated Git worktree and the inventory to locate data/results; old embedded absolute artifact paths may need explicit adjustment.

Current retained artifacts:

- `artifacts/benchmark/cross_subject_fold0_bn_control/`: common warmup checkpoint and active paired runs.
- `artifacts/benchmark/reference_constant_lr/`: old constant-LR metrics and fixed-subset predictions; weights are archived.
- `artifacts/benchmark/reference_cosine_partial/`: interrupted cosine-run logs/configuration; its step-100 checkpoint is retained under the current common directory.
- `artifacts/benchmark/mpnet/`, instruction embeddings and source audit files: current shared dependencies.
- `artifacts/models/`: shared downloaded model.
- `artifacts/ecg_r1/`: separate user ECG-R1 work.

Shared `datasets/`, environments, `laya/` and external third-party repositories were not removed or moved. Unrelated uncommitted ECG files and editor settings were not staged into the experiment commit. Historical scripts that had not been tracked, such as the canceled training plot helper, are preserved under `untracked_historical_scripts/` in the archive.

## Cleanup verification (2026-10-25)

The archive contains **383 path entries, 17,774 files and 265.56 GiB of logical file contents**. These are archive entries, not 383 independent experiments. No archived checkpoint, dataset, log or cache was deleted.

The superseded tracked implementations were removed in `55008f1`, after preserving the prior state in `521f865`. A final pass also moved 255 orphaned bytecode files into the archive's `retired_bytecode/` tree, including the otherwise empty `eeg_probes/`, `eeg_tasks/` and `tusz_sft/` packages. Current source files and their runtime caches remain in place. The 16 files watched by the active training controller are unchanged.

The working tree retains the current benchmark implementation, matched BN controls, necessary tests and shared dependencies. Independent, uncommitted ECG-R1/fNIRS work, editor settings, environments, third-party installations and datasets remain untouched. `artifacts/` now has only `benchmark/`, `models/`, `ecg_r1/` and the archive index. The compact historical benchmark metrics in `reference_*` are retained solely for the current comparison.

## Historical result entry points

[Current results and comparisons](RESULTS.md) remain the main result record. The following historical reports are preserved verbatim; their figures belong to their original protocols and are not directly comparable with the current benchmark. Historical JSON/JSONL metrics and checkpoint locations are preserved in the same archive and can be located through `index.json`. Reports may contain old absolute paths; use the archive index to resolve those paths.

| Historical experiment | Preserved report |
| --- | --- |
| `channel_prototypes_v5` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/channel_prototypes_v5/RESULTS.md) |
| `domain_validation_v11` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/domain_validation_v11/RESULTS.md) |
| `eeg_decision_experiments` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/eeg_decision_experiments/RESULTS.md) |
| `feature_probes` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/feature_probes/RESULTS.md) |
| `full_sources_v12_evaluation` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/full_sources_v12_evaluation/RESULTS.md) |
| `jev_interface_diagnostic` | [REPORT.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/jev_interface_diagnostic/REPORT.md) |
| `jev_paired_qa/readout_evaluation` | [REPORT.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/jev_paired_qa/readout_evaluation/REPORT.md) |
| `mi_interface_v2` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/mi_interface_v2/RESULTS.md) |
| `mi_zero_shot_v1` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/mi_zero_shot_v1/RESULTS.md) |
| `multitask_prototypes_v1` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/multitask_prototypes_v1/RESULTS.md) |
| `reve_only_multi/train_20261015_1404` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/reve_only_multi/train_20261015_1404/RESULTS.md) |
| `reve_stage_diagnostic/run_20261015_1444` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/reve_stage_diagnostic/run_20261015_1444/RESULTS.md) |
| `reve_tail_v8` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/reve_tail_v8/RESULTS.md) |
| `spatial_prototypes_v6` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/spatial_prototypes_v6/RESULTS.md) |
| `spatial_repeats_v7` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/spatial_repeats_v7/RESULTS.md) |
| `spectral_fusion_v10` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/spectral_fusion_v10/RESULTS.md) |
| `task_readouts_v13_evaluation` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/task_readouts_v13_evaluation/RESULTS.md) |
| `task_subject_disentangle` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/task_subject_disentangle/RESULTS.md) |
| `trial_aggregation_v9` | [RESULTS.md](../../eegjev_archive/20261025_pre_bn_control/artifacts/trial_aggregation_v9/RESULTS.md) |
