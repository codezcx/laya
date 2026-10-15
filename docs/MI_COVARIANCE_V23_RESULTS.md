# MI relative-amplitude diagnostic v23

Both source-only models completed whole-domain validation on the audited
37,432-trial source pool:33,272 training and4,160 validation trials. All four
feature shards passed manifest/hash, finite-array and exact index coverage checks.
No original source test was used. All fits completed without warnings.
[Protocol](MI_COVARIANCE_V23_PROTOCOL.md).

| Representation | Selected C | Held Cho2017 % | Held Dreyer2023 % | Held Lee2019_MI % | Domain mean % | Within-source mean % |
|---|---:|---:|---:|---:|---:|---:|
| Correlation | 0.01 | 57.80 | 65.28 | 56.90 | 59.99 | 60.06 |
| Relative covariance | 0.001 | 53.00 | 71.16 | 60.30 | 61.49 | 61.92 |

Held-domain means use equal dataset weight and subject-macro accuracy within
each dataset. Each fold excludes the entire held dataset from fitting and
standardization; only its existing validation subjects are scored. C is selected
on the mean of all three folds, so these scores are development evidence rather
than independent estimates of a selected model's generalization.

Preserving inter-electrode variance improves the mean by1.49points, but changes
individual held-domain scores by−4.80,+5.88,+3.40points. It is not a uniformly
better representation. Final models were refit on all source training subjects
using the selected C values, with both model hashes frozen before2b access.
These are conventional signal baselines, not language models or proof of text
alignment. No new tasks are introduced; emotion results remain unchanged.

## Exploratory2b evaluation

Both frozen models were evaluated on all2,840 original E trials from9subjects,
with identical feature extraction and no target fitting or adaptation. Source
and target file paths are disjoint. All model hashes matched before target read.

| Representation | Trial accuracy % | Subject-macro accuracy % | Shuffled subject-macro % |
|---|---:|---:|---:|
| Correlation | 53.27 | 53.27 | 47.29 |
| Relative covariance | 69.61 | 69.44 | 49.86 |

The paired subject-macro difference is+16.18percentage points, with a10,000-draw
subject bootstrap95% interval[+7.53,+25.80]. Eight of nine subjects improve;
subject7 decreases from70.00% to67.81%. The relative covariance model spans
56.25–85.00% subject accuracy, so performance remains heterogeneous.

This provides useful evidence that retaining spatial variance ratios matters
for this filter-bank representation on2b. It does not prove that the same gain
will transfer to REVE or a language interface, nor does it isolate the effect of
the earlier REVE broadband normalization: this diagnostic normalizes each band
separately. The models also have independently source-selected regularization.
No pretrained encoder is used here, so REVE pretraining exposure is irrelevant
to these two baselines; prior repeated2b development still limits interpretation.

Next: retain this conventional baseline and test a matched REVE input comparison
with common trial scaling versus per-channel scaling, identical filtering,
training coverage and domain-selection rules. Do not promote the language model
or declare universal zero-shot success from this signal-only result.

Artifacts: `artifacts/mi_covariance_v23/` contains feature shards and provenance;
`fits/complete.json` contains every fold/C score and final model hashes;
`external_2b/` contains the frozen manifest and subsequent external evaluation.
The2b evaluation is repeated-target exploratory evidence, not a fresh blind test.

Tests:3passed, covering common-gain invariance, electrode-gain sensitivity,
flat/nonfinite rejection and whole-domain/test exclusion. Runtime source audit
additionally verified all37,432 indices exactly once and subject-disjoint splits.
