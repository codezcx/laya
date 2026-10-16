# Subjective-rating auxiliary controls v28

All four2000-step runs completed on four GPUs with the fixed shared split,
initialization, batch sequence and architecture. All source-selected checkpoints
are step200; their SHA256 hashes were verified and frozen. No external target
was read. [Protocol](EMOTION_RATING_V28_PROTOCOL.md).

Held FACED videos:[2,5,9,10,15,18,22,23,26]. Training has17,912 windows; validation
has1,350SEED,2,250FACED unseen-video and4,750FACED seen-video windows. All FACED
validation subjects are absent from training; held videos are absent from every
training subject. Source tests remain excluded.

| Auxiliary supervision | SEED balanced % | FACED unseen-video balanced % | FACED seen-video balanced % | Selection mean % |
|---|---:|---:|---:|---:|
| None | 48.37 | 38.50 | 40.01 | 43.44 |
| Actual Valence | 48.67 | 38.73 | 39.71 | 43.70 |
| Within-class shuffled Valence | 49.04 | 38.93 | 39.64 | 43.99 |
| Class-mean Valence | 48.37 | 38.50 | 39.75 | 43.44 |

Selection mean weights SEED and unseen-video FACED equally. Actual ratings add
only0.26percentage points over pure classification, while within-class shuffling
adds0.55points. This single-seed source-development experiment does not establish
an improvement from individualized rating information. Do not interpret a small
positive point difference as successful subjective-semantic supervision. The
result does not prove subjective ratings are inherently useless: representation,
loss strength, temporal mismatch and source quality remain possible limitations.
The current fixed recipe is not supported as an improvement, so no target
evaluation is automatically scheduled for it.

Seen/unseen-video scores use different stimuli and are descriptive; their
difference is not a paired causal estimate of stimulus memorization. All27prior
rounds of source development also preclude treating this as a fresh independent
benchmark. This uses a spectral MLP, not REVE/Qwen, and does not validate a language
interface or task recognition. MI unchanged; the full goal remains unachieved.

Reproducibility: `artifacts/emotion_rating_v28/plan.json` fixes indices, hashes,
training-only rating normalizer and protocol;`targets.npz` stores auxiliary
targets;arm directories retain losses, all validation checkpoints' metrics,
normalizers and best weights. `source_freeze.json` records all four model hashes.
The split test verifies held subjects/videos and source-test exclusion.
