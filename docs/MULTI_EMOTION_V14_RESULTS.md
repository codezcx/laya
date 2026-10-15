# FACED multi-source experiment v14 results

All four predeclared arms completed 3,000 steps (seed 42). Frozen REVE features; a shared spatial classifier is trained. Qwen supplies offline text prototypes only for the text arm. This is known-task classification, not automatic task recognition or unseen-task zero-shot.

FACED: fixed 16 training and 8 validation participants, 6,720 windows; full integrity audit passed. HEOR/HEOL auxiliary EOG channels are excluded where present. MI training remains Cho2017 + Dreyer2023 + Lee2019. All arms share SEED + FACED source validation; checkpoint selection uses macro-task balanced accuracy.

| Emotion training / head | Source macro balanced % | SEED val balanced % | FACED val balanced % | SEED-VII subject balanced % | 2b subject balanced % |
|---|---:|---:|---:|---:|---:|
| seed_only | 50.66 | 42.30 | 37.19 | 38.10 | 58.81 |
| seed_faced | 51.45 | 44.52 | 39.86 | 38.79 | 58.66 |
| faced_only | 48.47 | 34.81 | 37.26 | 34.84 | 58.80 |
| seed_faced_text | 50.68 | 41.85 | 38.19 | 37.91 | 59.06 |

Adding FACED changes SEED-VII from 38.10% to 38.79%, a small single-seed difference, insufficient evidence of robust improvement. The text-prototype arm does not beat the joint ordinary classifier on emotion. FACED-only transfer is weak (34.84%). MI remains around 59%. These results do not establish the intended general-purpose zero-shot capability.

All frozen checkpoint hashes were verified before external evaluation. Targets were previously inspected, so these are exploratory repeated-target results, not a new blind test. REVE pretraining includes 2b. No checkpoint was selected using these target scores.

Artifacts: `artifacts/multi_emotion_v14_evaluation/external_results.json`, `all_frozen.json`; each run contains `best_metrics.json`, `sampling_audit.json`, `validation.jsonl`, and `complete.json`.

Next scientific priority: task-identification evaluation must separate acquisition/dataset recognition from EEG semantics. A task-level contrastive objective needs multiple datasets per task and dataset-held-out task prediction, with the ordinary classifier retained as a control. Do not infer task recognition from the masked known-task scores above.
