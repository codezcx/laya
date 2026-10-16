# Emotion relative spatial power v26: source results

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
`artifacts/emotion_power_v26/source_freeze.json`. External evaluation has not
yet run. The original controller retains its failed audit state; successful
recovery is explicitly recorded in
`artifacts/emotion_power_v26_controller/precision_recovery_complete.json`.
MI is unchanged. The universal zero-shot objective remains unachieved.
