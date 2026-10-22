# Current joint MI/emotion run

Implementation and protocol: [PROTOCOL.md](PROTOCOL.md). GPU2–3 only. Preparation completed: MI49,193 trials, SEED22,734 windows/405 parent trials, SEED-VII41,844 windows/960 parent trials in training. Validation is fixed and subject-disjoint within each dataset. Both two-GPU smoke checks passed (contrast backward/validation and JEV mixed-choice backward/paired validation); 12 unit/distributed tests passed. Contrast completed800steps. The first JEV run was interrupted by node loss at step173, before any checkpoint was saved. A fresh JEV run using the selected contrast encoder is prepared under `jev_restart`; training microbatch4, validation microbatch8, length-sorted validation. Smoke metrics are execution checks, not model-quality results.

- Pipeline: `artifacts/mi_emotion/pipeline.json`
- Contrast: `artifacts/mi_emotion/contrast/metrics.jsonl`
- JEV: `artifacts/mi_emotion/jev_restart/metrics.jsonl`
- Best validation and checkpoint: each stage's `best.json` and `best.pt`
- Data inventory/splits: `datasets/mi_emotion/complete.json`

Previous MI-only experiments and results remain in Git and their original ignored artifact directories. They are not joint-task results. The prior two arms stopped at logged step1200; saved validation ended at1000. At step600, normal-option source validation accuracy was74.55% for CE preconditioning and77.19% for text contrast preconditioning. These are subject-disjoint source-validation accuracies, not training or unseen-dataset test results.
