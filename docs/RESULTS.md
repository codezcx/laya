# TUAB-only results

The user replaced the MI/emotion experiment with TUAB normal/abnormal classification. Preprocessing and four-GPU frozen REVE extraction passed. Four-GPU two-step smoke passed and the formal725-step JEV run has launched. No formal JEV validation improvement is claimed yet.

Run status: `artifacts/tuab_jev/status.json`. Record counts/exclusions: `datasets/tuab/audit.json`. Validation and final test probabilities: `artifacts/tuab_jev/run/validation_*.json` and `test.json`. The frozen REVE logistic control is under `artifacts/tuab_jev/probe/`.

Report ordinary record accuracy first, then BA and normal/abnormal recall. Model selection uses validation BA; do not select checkpoints using test scores. The old MI/emotion results are archived in Git at `f83edcb`, and their ignored experiment outputs remain on disk.

## Audited data and launch

| Split | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Train | 1165 | 1149 | 2314 |
| Validation | 206 | 197 | 403 |
| Test | 150 | 126 | 276 |

All2993records retained; zero exclusions. Subjects disjoint across splits. Five epochs at up to16records/batch gives725optimization steps. Eleven unit tests passed, including equivalence of two-pass record gradients to full-bag autograd. Four-GPU smoke passed: finite gradients, synchronized trainable parameter hashes, and matching two-pass forward logits. Smoke uses only8records for execution checks, not a scientific validation estimate.

## Frozen REVE simple-head validation control

The train-only fitted StandardScaler+logistic regression selected C=0.01 using the403-record source validation set: accuracy83.13%, BA82.99%, normal recall89.32%, abnormal recall76.65%, AUROC0.9064. Confusion matrix (true rows normal/abnormal): `[[184,22],[46,151]]`. This is not a JEV result and not the held-out test result. Test is deferred until after formal JEV training. Evidence: `artifacts/tuab_jev/probe/validation.json`.
