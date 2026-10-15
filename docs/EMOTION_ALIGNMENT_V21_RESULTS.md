# Cross-source REVE tail adaptation v21 results

Four source-trained arms and one joint-source spectral control completed. Evaluation used the fixed SEED-VII Happy/Neutral/Sad subset (20subjects,6400windows), without target fitting or model choice. [Protocol](EMOTION_ALIGNMENT_V21_PROTOCOL.md).

| Arm | Source validation balanced % | Selected step | SEED-VII subject balanced % (95% CI) |
|---|---:|---:|---:|
| frozen_ce | 46.08 | 600 | 40.75 (37.69–43.86) |
| tail_ce | 45.48 | 600 | 40.71 (37.50–43.95) |
| tail_align | 45.64 | 600 | 41.08 (37.82–44.47) |
| tail_align_spectral | 45.74 | 500 | 40.87 (38.14–43.81) |
| spectral_control | 44.68 | C=0.01 | 43.72 (40.57–46.94) |

## Interpretation

Updating the last two REVE blocks with ordinary CE alone gives essentially the same transfer score as freezing them. Cross-source supervised alignment adds only0.33percentage points in this single-seed run; adding the spectral auxiliary loss adds0.12points relative to frozen_ce. These differences are insufficient to claim a useful adaptation gain. The jointly trained linear spectral control remains higher at43.72%. No target-best model is promoted.

This is a genuine encoder adaptation experiment, not just another decision-head change. `tail_update_audit.json` verifies zero weight change in frozen_ce and changes in12 tail tensors for each adapted checkpoint. Source-prefix extraction replays the original final two blocks within1e-5 in fp32, then caches fp16 prefixes identically for all arms. The adapter inputs are the same30-channel montage.

The new objectives are supervised by emotion class identity and measured band fractions; they are not report-level semantic alignment or proof of language-model understanding. Only the emotion branch was changed. Known-task emotion performance remains insufficient for the universal zero-shot objective, and MI results remain unchanged. No stress/arithmetic tasks were added.

## Limits and next decision

SEED-VII was exposed in earlier development, so this is exploratory repeated-target evaluation, not a new blind test. The happy/neutral/sad mapping is not full seven-class SEED-VII recognition. Source selection uses subject validation within the two training datasets; it may still favor acquisition-specific features. Bootstrap intervals measure subject variation, not seed uncertainty.

The evidence does not justify scaling this exact class-only alignment recipe to a larger language model. Further work should first change the available source-domain supervision or training-domain validation design, while retaining the spectral baseline. The small positive point difference here is not a sufficient basis for claiming a breakthrough or expanding tasks.

## Reproducibility

`artifacts/emotion_alignment_v21/` retains prefix manifests/replay errors, all source loss histories and checkpoints, training-only spectral normalizers, exact tail weight-delta audit, source-only spectral-control C sweep and target extraction hashes. `external_seed_vii/` contains all logits, subject metrics, shuffled controls for neural arms and paired differences.

Tests: `tests/test_emotion_alignment.py`,2passed (same-class cross-source pairing and missing-positive rejection). Source training and external evaluation controllers both completed successfully.
