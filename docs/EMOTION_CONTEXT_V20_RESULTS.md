# Emotion frequency/context factorial v20 results

Four extraction shards and eight source-selected fits completed. Both preprocessing tests passed. The30Hz/3s control reproduces every v19 as-recorded feature exactly (max absolute difference0). All eight model hashes froze before held-dataset evaluation. [Protocol](EMOTION_CONTEXT_V20_PROTOCOL.md).

| Band / contiguous duration | SEED → FACED balanced % (95% CI) | FACED → SEED balanced % (95% CI) |
|---|---:|---:|
| hz30_s3 | 38.15 (36.00–40.52) | 43.41 (40.28–46.93) |
| hz45_s3 | 37.25 (35.09–39.54) | 38.80 (36.59–41.31) |
| hz30_s9 | 35.79 (33.66–38.30) | 40.56 (38.11–43.11) |
| hz45_s9 | 37.12 (35.23–39.00) | 42.41 (39.33–45.44) |

## Finding

Neither extending the upper band to45Hz nor extending contiguous context to9seconds improves the held-dataset point estimates over the matched30Hz/3s control in either direction. The original control remains38.15% for SEED→FACED and43.41% for FACED→SEED. This does not establish that higher-frequency or temporal features are inherently useless; the test uses log-band fractions and linear classification, not a temporal sequence encoder. Wider-band normalization also changes the denominator of every fraction.

Nine-second windows remain within their original trial and contain the original3seconds. This is longer input, not more independent trials, and must not be described as3second inference. Some long windows overlap; all subject windows stay in one split and bootstrap uncertainty resamples subjects.

## Consequence for the next iteration

The tested inexpensive readout/preprocessing variations (fusion, average reference, band extension, contiguous context extension) have not improved the established frequency baseline. Stop treating these choices as the principal route to the goal. A subsequent change should alter representation learning or the source-domain training/selection scheme and preserve the strongest unchanged baseline. Improving source validation alone is insufficient, as previous experiments repeatedly showed.

Known-task emotion development transfer remains incomplete as a universal zero-shot system. The experiment did not update MI, evaluate2b/SEED-VII or add arithmetic/stress. Repeatedly examined development datasets and one source partition limit inference; no fresh blind-test claim.

## Artifacts

`artifacts/emotion_context_v20/features/` contains source hashes, exact per-sample per-mode bounds, channel order and recording metadata. `control_reproduction.json` stores the exact replay check. Each mode/source retains source-only C sweeps, selected model/hash; `evaluation/` retains all eight predictions, shuffled controls, subject scores and paired differences.
