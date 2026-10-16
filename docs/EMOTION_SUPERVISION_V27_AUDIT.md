# Emotion supervision audit v27

This source-only audit examines the semantics and coverage of existing labels
before another model change. No model was fit, no target rating/label file was
read, and no subject, trial or window was relabeled or excluded.

## Verified local evidence

All98 FACED development subjects have28 stimulus events and valid numeric
Valence ratings in[0,7]:2,044 training and700 validation trials. Existing EEG
window labels and onset times match their stimulus events exactly. The current
classifier is supervised by `binary_label`, the video's intended valence,
not the participant's subjective Valence response.

| Source split / stimulus class | Trials | Mean Valence | 10th / 50th / 90th percentile |
|---|---:|---:|---|
| Train / negative | 876 | 1.80 | 0.06 / 1.42 / 3.75 |
| Train / neutral | 292 | 3.19 | 1.08 / 3.48 / 4.40 |
| Train / positive | 876 | 4.49 | 2.44 / 4.54 / 6.58 |
| Validation / negative | 300 | 1.44 | 0.03 / 0.91 / 3.51 |
| Validation / neutral | 100 | 3.04 | 0.46 / 3.48 / 4.06 |
| Validation / positive | 300 | 4.30 | 2.14 / 4.28 / 6.64 |

Seven of73 training subjects have mean positive-stimulus Valence no greater than
their negative-stimulus mean; none of25 validation subjects do. This describes
heterogeneity, not annotation error or a justification to remove people. Ratings
may reflect experience, response style, recording/conversion issues, or multiple
causes. Subject IDs and full descriptive statistics are retained in the artifact.

All28 FACED video IDs occur in both training and validation subjects. Thus
ordinary held-subject validation does not test unseen videos. SEED has all15
trial indices in both splits; those indices alone do not establish identical
clip identity across its sessions, so no stronger clip-overlap claim is made.
SEED's local readme specifies fixed trial-level categories. These audits do not
prove a model uses stimulus shortcuts, only that current splits do not exclude
that explanation for within-source scores.

## Provenance and interpretation

FACED's original data descriptor documents stimulus induction and subjective
ratings, including valence and arousal:
[Chen et al., Scientific Data (2023)](https://www.nature.com/articles/s41597-023-02650-w).
The pinned NEMAR README and actual event TSV columns provide local conversion
context. The README contains inconsistent author/article metadata, so it is not
used as authoritative bibliographic attribution. Rating semantics and conversion
provenance need verification before interpreting any disagreement physiologically.

`scripts/audit_emotion_supervision_v27.py` joins only the unchanged source cohort
and records SHA256 hashes for all98 event files. Output lives in
`artifacts/emotion_supervision_v27/{audit.json,ratings.jsonl}`. It preserves the
fixed73/25 subject split, the video identity and continuous rating separately.

## Next experimentally useful change

Retain stimulus classification as the primary comparable task. Test continuous
subjective-rating prediction as auxiliary source supervision, against an otherwise
matched classification-only model. Do not invent negative/neutral/positive rating
thresholds or drop disagreeing subjects. Fix model/loss choices on source data;
include a source holdout excluding video IDs as well as subjects to assess
stimulus-specific generalization. Preserve a plain signal baseline and a shuffled
auxiliary-label control to distinguish useful supervision from regularization.

This is a proposed next experiment, not a measured improvement. Its data lineage
and exact split/loss protocol must be locked before training. Do not read SEED-VII
continuous labels to optimize this design. MI results remain unchanged and the
universal zero-shot goal remains unachieved.
