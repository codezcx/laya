# Emotion relative spatial power v26

Both arms completed source fitting without convergence warnings after the
precision audit described in the[protocol](EMOTION_POWER_V26_PROTOCOL.md).
All32,827 windows were retained; neither arm uses target statistics.

| Representation | C | FACED→SEED validation balanced % | SEED→FACED validation balanced % | Mean % |
|---|---:|---:|---:|---:|
| Per-channel fractions | 0.01 | 39.11 | 37.41 | 38.26 |
| Joint power fractions | 0.01 | 41.70 | 37.19 | 39.45 |

Scores average subject-level balanced accuracy within each held dataset. Each
fold fits only the other dataset's training subjects; C is selected by the mean
of the two held-domain validation scores. The1.19point mean improvement comes
from FACED→SEED, with a small decrease in the reverse direction. This is not
consistent evidence of a broadly better representation or independent testing.

Raw float64 control features differ fromv22 in214 windows of FACED subject30.
All280 subject30 windows were replayed to attribute the discrepancy to the old
normalized-float32 representation; no clipping occurred. Bothv26 arms use the
same raw float64 power computation. No subject/window was removed. Original
failed audit/logs remain preserved; exact exception checks and evidence SHA256
are recorded in the new feature audit.

Final models were refit on both source training sets and hashes verified in
`artifacts/emotion_power_v26/source_freeze.json`. The original controller retains its failed audit state; successful
recovery is explicitly recorded in
`artifacts/emotion_power_v26_controller/precision_recovery_complete.json`.
MI is unchanged. The universal zero-shot objective remains unachieved.

## Exploratory SEED-VII evaluation

Both frozen models were evaluated on exactly the same6,400 HNS windows from
20subjects. All four control feature shards exactly reproduced the earlier
float64 spectral cache (maximum difference0), and model/manifest/feature hashes
were verified before prediction. No target fitting or selection was performed.

| Representation | Accuracy % | Subject-balanced % | Balanced95% CI | Shuffled balanced % |
|---|---:|---:|---:|---:|
| Per-channel fractions | 43.78 | 43.68 | 40.67–46.72 | 34.27 |
| Joint power fractions | 43.25 | 43.05 | 39.66–46.53 | 32.65 |

The joint-minus-per-channel difference is−0.63percentage points, paired subject
bootstrap95% CI[−3.34,+1.84]. The modest source-selection gain did not translate
into external improvement. This experiment does not support transferring the MI
relative-amplitude change directly to emotion. Keep both results rather than
choosing a new target-dependent normalization. The control differs slightly
fromv22 because its FACED training features use raw float64 power.

SEED-VII is a repeatedly exposed exploratory target and only Happy/Neutral/Sad
are evaluated, not its full7classes. These classifiers are not language models
and do not infer task identity. Current evidence favors retaining the established
emotion spectral baseline while revisiting source/task supervision rather than
continuing to claim success from small validation-only gains.

`eeg_tasks/emotion_power_external_v26.py` and
`scripts/evaluate_emotion_power_v26.py` implement extraction/evaluation. All
predictions, cache hashes, per-subject metrics and paired differences remain in
`artifacts/emotion_power_v26/external_seed_vii/`; its separate controller completed.
