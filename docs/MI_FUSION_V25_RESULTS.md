# MI probability complementarity v25

Six held-domain source models completed without warnings. Source feature caches
passed hash and index audits, and the covariance/REVE manifests matched exactly.
All source-selected C values stayed fixed; no target values selected the mixing
weight. [Protocol](MI_FUSION_V25_PROTOCOL.md).

| REVE probability weight | Held-source mean subject accuracy % |
|---|---:|
| 0 | 61.49 |
| 0.25 | 62.05 |
| 0.50 | 61.13 |
| 0.75 | 60.13 |
| 1 | 59.81 |

Select alpha=.25:75% relative-covariance probability plus25% common-scale REVE
probability. The source-development improvement over covariance alone is only
0.57points. This selection reuses development folds previously used for C and
is not an unbiased nested-CV estimate. The two full-source model hashes and the
chosen alpha are fixed before external evaluation. No alternate target weights
will be swept or selected.

Source probabilities, every candidate score and model provenance are retained
in`artifacts/mi_fusion_v25/`. This remains known-task classification; emotion and
language-interface performance are unchanged. Universal zero-shot is unproven.

## Exploratory2b results

| Model | Trial accuracy % | Subject-macro accuracy % |
|---|---:|---:|
| Covariance alone | 69.61 | 69.44 |
| REVE alone | 63.27 | 63.02 |
| Source-selected mixture | 68.77 | 68.52 |

The mixture is0.92percentage points lower in subject-macro accuracy than the
covariance baseline; paired subject bootstrap95% CI[−3.05,+0.97]. Thus the small
source-development gain did not translate into a clear target improvement.
This does not prove all fusion mechanisms fail, but gives no evidence to deploy
this mixture as an improvement. Preserve the covariance baseline and report the
selected mixture as a negative result; do not pick another alpha on this target.

Both single-model predictions exactly reproduced theirv23/v24 saved predictions
on all2,840 trials. Frozen full-source model hashes and the target manifest hash
were verified. All three probability arrays and paired statistics are saved in
`external_2b/`. Evaluation is repeated-target exploratory, not a new blind test;
REVE pretraining includes2b. No target adaptation was performed.
