# FACED source-scale controls v22

All 98 source development subjects completed acquisition and integrity checks:
73 training and 25 validation subjects. The 25 reserved test subjects were not
acquired. Audit retained 27,427 windows and explicitly quarantined 13 flat-PO3
windows from training subject100. Original failure status and controller logs
remain archived under `artifacts/faced_v22_reconciliation/original_controller`
and `original_cohort_status.json`. [Protocol](FACED_FULL_V22_PROTOCOL.md).

## Matched source validation

Both training scales use the identical 8,350 validation windows, verified by
ordered-row SHA256: SEED 1,350 plus FACED 7,000. Pilot training has 8,530 windows;
full training has 24,477 after the quality exclusions. SEED training is unchanged.
All four source-selected models use C=0.01; no convergence warnings occurred.

| FACED training | Spectral representation | SEED balanced % | FACED balanced % | Dataset mean % |
|---|---|---:|---:|---:|
| 16 subjects | Full common30 | 48.07 | 38.80 | 43.44 |
| 73 subjects | Full common30 | 50.59 | 39.13 | 44.86 |
| 16 subjects | Spatially centered common30 | 48.81 | 38.24 | 43.53 |
| 73 subjects | Spatially centered common30 | 50.52 | 38.90 | 44.71 |

Each dataset score averages subject-level balanced accuracy. Expansion adds
1.43 points for full features and 1.18 points for centered features. Much of the
gain is on SEED validation; FACED itself improves only 0.33/0.66 points. This is
modest source-validation evidence, not proof of improved external transfer or
automatic task recognition. These are spectral classifiers; no LLM is used.

## Next evaluation boundary

All four checkpoint hashes and their shared validation-row hash were verified
and recorded in `artifacts/faced_scale_v22/source_freeze.json` before further
target evaluation. At this report's creation no v22 target scores have been
computed. Evaluate all four frozen arms on the existing SEED-VII HNS protocol;
report paired subject differences for full versus pilot within each feature
representation, without selecting a model using target performance. This is a
repeated-target exploratory evaluation, not a fresh blind test. MI is unchanged;
the overall zero-shot goal remains unachieved and no additional task is enabled.

Artifacts include source data manifest, feature cache, every C sweep, model
hashes, and completed controller state. Code and this summary are tracked in
Git; data, model files and detailed runtime records remain ignored artifacts.
