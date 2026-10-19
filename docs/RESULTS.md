# TUAB-only results

The user replaced the MI/emotion experiment with TUAB normal/abnormal classification. Implementation is prepared; no TUAB training result is claimed yet.

Run status: `artifacts/tuab_jev/status.json`. Record counts/exclusions: `datasets/tuab/audit.json`. Validation and final test probabilities: `artifacts/tuab_jev/run/validation_*.json` and `test.json`. The frozen REVE logistic control is under `artifacts/tuab_jev/probe/`.

Report ordinary record accuracy first, then BA and normal/abnormal recall. Model selection uses validation BA; do not select checkpoints using test scores. The old MI/emotion results are archived in Git at `f83edcb`, and their ignored experiment outputs remain on disk.
