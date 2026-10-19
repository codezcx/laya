# TUAB-only results

The user replaced the MI/emotion experiment with TUAB normal/abnormal classification. Pipeline launched; preprocessing passed and four-GPU frozen REVE extraction is running. No JEV training result is claimed yet.

Run status: `artifacts/tuab_jev/status.json`. Record counts/exclusions: `datasets/tuab/audit.json`. Validation and final test probabilities: `artifacts/tuab_jev/run/validation_*.json` and `test.json`. The frozen REVE logistic control is under `artifacts/tuab_jev/probe/`.

Report ordinary record accuracy first, then BA and normal/abnormal recall. Model selection uses validation BA; do not select checkpoints using test scores. The old MI/emotion results are archived in Git at `f83edcb`, and their ignored experiment outputs remain on disk.

## Audited data and launch

| Split | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Train | 1165 | 1149 | 2314 |
| Validation | 206 | 197 | 403 |
| Test | 150 | 126 | 276 |

All2993records retained; zero exclusions. Subjects disjoint across splits. Five epochs at up to16records/batch gives725optimization steps. Eleven unit tests passed, including equivalence of two-pass record gradients to full-bag autograd. Four-GPU smoke remains required before formal training.
