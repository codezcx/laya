# Matched-montage REVE / spectral fusion v17 results

All four arms completed both transfer directions (eight 3,000-step fits). Four extraction shards cover the same 12,120 SEED/FACED windows with exactly 30 common channels before REVE encoding. All eight source-selected checkpoint hashes were checked before held-dataset evaluation. Protocol: [v17](EMOTION_FUSION_V17_PROTOCOL.md).

| Source → held dataset | Arm | Source validation balanced % | Held balanced % (subject 95% CI) | Shuffled % |
|---|---|---:|---:|---:|
| SEED → FACED | spectral | 45.11 | 36.62 (34.70–38.65) | 33.24 |
| FACED-NEMAR → SEED | spectral | 40.87 | 41.93 (39.11–44.94) | 33.54 |
| SEED → FACED | reve | 48.44 | 36.86 (35.50–38.40) | 33.47 |
| FACED-NEMAR → SEED | reve | 39.69 | 38.74 (36.30–41.11) | 32.98 |
| SEED → FACED | fusion | 46.89 | 36.12 (34.48–37.85) | 33.72 |
| FACED-NEMAR → SEED | fusion | 38.44 | 38.33 (36.78–39.91) | 33.07 |
| SEED → FACED | fusion_text | 47.41 | 36.46 (34.75–38.23) | 33.66 |
| FACED-NEMAR → SEED | fusion_text | 39.72 | 38.11 (36.00–40.19) | 32.65 |

Metrics are mean subject balanced accuracy. Chance is 33.33%. Held cohorts are 24 FACED and 12 SEED development subjects, not newly blinded test sets. No 2b or SEED-VII evaluation occurred. Source validation selects checkpoints only, not held-dataset scores.

## Conclusion

Simple feature concatenation does not improve held-dataset emotion transfer. FACED → SEED spectral MLP reaches 41.93%, whereas fusion reaches 38.33% and fusion with text prototypes 38.11%. In the reverse direction all arms remain around 36–37%. The older v16 regularized linear full-channel spectral baseline remains stronger (43.41% / 38.16% in those directions; spatial centering reached44.17% in FACED → SEED).

Thus, useful frequency information is not automatically preserved by a larger fused neural readout. The present evidence does not support adding this concatenation or fixed text head to the production path. It also does not isolate whether frozen REVE, its temporal pooling, source selection, or optimization is responsible. These are known-task diagnostics; the requested unified zero-shot system remains incomplete.

A next representation experiment should preserve the established spectral decision directly as a control (e.g. a source-validated residual or late-fusion path) rather than making it compete with 15,360 latent features in a single MLP. Any encoder fine-tuning must demonstrate improvements over the simple frequency baseline on held datasets, not only lower training loss.

## Audit

- `features/{0..3}.done.json`: channel order, source manifest hash and REVE checkpoint; joined indices must cover every row exactly once.
- Each arm/source contains source-only standardizer buffers in `best.pt`, selected metrics, full validation history, parameter count and completion hash.
- `evaluation/frozen.json`, `results.json`, saved logits and `paired_differences.json` retain all eight results and comparisons.
- Tests: `tests/test_emotion_fusion.py`, 2 passed. Target changes do not alter source standardization; both output heads propagate gradients to their signal input.
- Fixed seed42 only; bootstrap intervals measure subject variation, not seed variability. Head parameter counts differ; no claim of parameter-matched text benefit.
