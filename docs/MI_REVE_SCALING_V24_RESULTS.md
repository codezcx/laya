# Frozen REVE scaling v24

All37,432 source trials were extracted for both scaling arms on four GPUs.
Every per-channel input hash matchedv12 and the maximum fp16 feature difference
was0 across all four shards. Per-channel scaling clipped0 values; common trial
scaling clipped7. [Protocol](MI_REVE_SCALING_V24_PROTOCOL.md).

## Source selection and convergence repair

The initial C=10 runs reached the2,000-iteration limit in the held Cho2017 and
Lee2019_MI folds of both arms. Those four folds were recomputed with a10,000
limit before target evaluation. All repaired folds and final refits completed
without warnings. Initial artifacts remain in`fits/`; authoritative source
results are in`fits_converged/`, linked to the original result SHA256.

Repair changed common scaling's selected C from10 to0.01. Per-channel C remains
0.1. Thus the initial apparent C=10 advantage was not reliable enough to use.

| Scaling | Selected C | Held-domain mean accuracy % | Within-source validation mean % |
|---|---:|---:|---:|
| Per channel | 0.1 | 59.48 | 63.07 |
| Common trial RMS | 0.01 | 59.81 | 62.31 |

Selection uses equal weight over three held datasets, each scored by subject
macro accuracy. Source selection improvement is only0.33percentage points;
within-source validation is slightly lower with common scaling. This does not
establish the large relative-covariance gain fromv23 for REVE.

Both converged final checkpoints are frozen before target access. Evaluation
checks every source fold for warnings, validates model hashes, and rejects
source/target file overlap. The existing2b E-trial manifest has2,840 trials.
Any target result is exploratory:2b has prior development exposure and is in
REVE pretraining. No target fitting or selection occurs. Emotion is unchanged;
the overall universal zero-shot objective remains unachieved.

## Exploratory2b results

| Scaling | Trial accuracy % | Subject macro accuracy % | Shuffled subject macro % |
|---|---:|---:|---:|
| Per channel | 60.56 | 60.33 | 49.29 |
| Common trial RMS | 63.27 | 63.02 | 49.33 |

Common scaling improves subject-macro accuracy by2.68points, paired subject
bootstrap95% CI[+0.82,+4.86] using10,000 draws. Both models used exactly the same
2,840 E trials across9subjects, identical filtering/encoder/pooling, and
source-only independently selected C. This supports retaining spatial variance
ratios in the REVE input, but does not isolate scaling from regularization
selection, demonstrate language understanding, or establish performance on a
fresh unseen target. The conventional relative-covariancev23 baseline remains
higher at69.61% trial accuracy; its filter bank and representation differ.

All predictions and fp16 target feature caches are retained under
`artifacts/mi_reve_scaling_v24/external_2b/`, alongside frozen model hashes and
target manifest SHA256. Source tests remain untouched. Next work should preserve
the stronger conventional baseline and determine whether learned representations
provide complementary information under source-only selection, rather than
assuming REVE or a larger language decoder is necessarily superior.
