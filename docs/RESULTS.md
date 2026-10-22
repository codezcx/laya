# Current joint MI/emotion run

Implementation and protocol: [PROTOCOL.md](PROTOCOL.md). GPU2–3 only. Preparation completed: MI49,193 trials, SEED22,734 windows/405 parent trials, SEED-VII41,844 windows/960 parent trials in training. Validation is fixed and subject-disjoint within each dataset. No production validation result yet at launch preparation.

- Pipeline: `artifacts/mi_emotion/pipeline.json`
- Contrast: `artifacts/mi_emotion/contrast/metrics.jsonl`
- JEV: `artifacts/mi_emotion/jev/metrics.jsonl`
- Best validation and checkpoint: each stage's `best.json` and `best.pt`
- Data inventory/splits: `datasets/mi_emotion/complete.json`

Previous MI-only experiments and results remain in Git and their original ignored artifact directories. They are not joint-task results. The prior two arms stopped at logged step1200; saved validation ended at1000. At step600, normal-option source validation accuracy was74.55% for CE preconditioning and77.19% for text contrast preconditioning. These are subject-disjoint source-validation accuracies, not training or unseen-dataset test results.
