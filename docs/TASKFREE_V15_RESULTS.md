# Task-free common-channel transfer v15 results

All four runs completed 3,000 steps with seed42; checkpoints were selected using only source validation and hash-verified before held-dataset evaluation. See [protocol](TASKFREE_V15_PROTOCOL.md). Input is C3/Cz/C4 for every dataset. Neither forward nor loss receives task ID. Qwen is used only for fixed offline text prototypes.

| Held-out dataset | Head | Task accuracy % | Joint class balanced % | Oracle-task class balanced % | Shuffled joint balanced % |
|---|---|---:|---:|---:|---:|
| Lee2019_MI | ordinary | 99.77 | 56.70 | 56.85 | 50.20 |
| SEED | ordinary | 99.54 | 33.19 | 33.35 | 33.04 |
| Lee2019_MI | text | 99.86 | 56.97 | 57.04 | 50.65 |
| SEED | text | 99.46 | 33.04 | 33.28 | 33.09 |
| FACED-NEMAR | ordinary | 86.93 | 29.31 | 33.66 | 29.98 |
| Lee2019_MI | ordinary | 100.00 | 56.76 | 56.76 | 48.95 |
| FACED-NEMAR | text | 84.20 | 27.78 | 32.79 | 28.99 |
| Lee2019_MI | text | 100.00 | 56.26 | 56.26 | 48.85 |

Metrics are averaged over subjects; class-balanced scores average recall over each subject’s true classes. Task accuracy uses summed probabilities for the two task groups. Joint prediction is the argmax across all five classes. Oracle-task accuracy supplies the true task for evaluation only. Shuffling permutes predictions within each held dataset, retaining dataset/task cues while breaking class correspondence.

## Interpretation

- Task identification is much easier than emotion classification. FACED-trained models identify SEED as emotion with approximately 99.5% accuracy, yet emotion class accuracy is at chance. High routing accuracy does not establish semantic understanding.
- SEED-trained models show task-routing errors on FACED (about 13–16%); even supplying the correct task only recovers chance-level emotion classification. Routing is therefore not the sole bottleneck.
- MI held-out Lee accuracy is approximately 56–57%, compared with shuffled results around 49–51%. This is modest development transfer, not the desired robust generalization.
- Fixed text prototypes do not deliver a meaningful advantage over the ordinary head. A task/class phrase alone has not supplied missing class-discriminative EEG information.
- The three-channel experiment is an explicit montage-control diagnostic. It does not replace the full-channel emotion result, and cannot establish that full-channel emotion learning is impossible.

## Limits and next action

All held-out datasets were used in prior development, so this is exclusion from the current training run, not a new blind benchmark. No 2b or SEED-VII evaluation occurred this round. Common channels remove a channel-count shortcut but leave hardware, protocol, and upstream preprocessing differences. Task labels denote the experimental task, not mutually exclusive underlying physiology.

The next priority is full-channel emotion representation and source-domain transfer, with signal baselines and ordinary heads retained. Adding more task-name supervision or expanding to arithmetic/stress would not resolve the failure demonstrated here. Any subsequent automatic-task design must report both routing and class performance and include acquisition controls.

## Reproducibility

- Data: `artifacts/taskfree_v15_data/complete.json` records 49,552 rows, source-manifest hashes, common montage, and feature checksum.
- Models/logs: `artifacts/taskfree_v15/<held-dataset>_<head>/`.
- Evaluation and fixed-shuffle controls: `artifacts/taskfree_v15/evaluation/results.json`.
- Subject bootstrap intervals (10,000 draws, seed2026): `artifacts/taskfree_v15/evaluation/confidence_intervals.json`. These intervals do not measure seed uncertainty.
- Tests: `tests/test_taskfree.py`, 2 passed; wrong-task outputs are not given masked credit and dataset holdouts exclude all their rows.
- Initial failed data-schema and dataset-name checks are preserved in `taskfree_v15_data_attempt1`, `taskfree_v15_data.log`, and `taskfree_v15_attempt1_dataset_name`. They failed before training, were repaired explicitly, and did not alter the planned splits.
