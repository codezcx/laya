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
