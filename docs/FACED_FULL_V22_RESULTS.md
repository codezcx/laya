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

## External evaluation

All four checkpoint hashes and their shared validation-row hash were verified
and recorded in `artifacts/faced_scale_v22/source_freeze.json` before further
target evaluation. All four frozen arms were then evaluated on the existing
SEED-VII HNS protocol (20 subjects, 6,400 windows), with no target fitting or
selection. This is repeated-target exploratory evaluation, not a fresh blind test.

| Training / features | Accuracy % | Subject balanced % | Balanced 95% CI |
|---|---:|---:|---:|
| Pilot / full | 44.09 | 43.72 | 40.57–46.94 |
| Full / full | 43.88 | 43.72 | 40.62–46.83 |
| Pilot / spatial | 42.86 | 42.67 | 39.67–45.84 |
| Full / spatial | 43.12 | 43.42 | 40.64–46.36 |

Paired subject bootstrap differences (full minus pilot) are +0.007 percentage
points for full features (95% CI −0.97 to +0.95) and +0.743 points for spatial
features (−0.56 to +1.99). Both intervals include zero. Shuffled-prediction
balanced controls range from33.48% to34.44%. These intervals quantify subject
sampling variability, not seed or repeated-development uncertainty.

The source-validation gain did not translate into a clear transfer improvement.
Increasing FACED from16 to73 training subjects alone is insufficient in this
representation. It does not establish that source diversity or richer signal
supervision would be ineffective, and does not justify claiming semantic
alignment. Avoid scaling the same recipe on the strength of source validation;
next development should address domain-level validation and supervision.
MI is unchanged; the overall zero-shot goal remains unachieved and no additional
task is enabled.

Artifacts include source data manifest, feature cache, every C sweep, model
hashes, and completed controller state. Code and this summary are tracked in
Git; data, model files and detailed runtime records remain ignored artifacts.
`scripts/evaluate_faced_scale_v22.py` verifies all source model and validation
hashes before loading target features, hashes the reused target cache, and saves
all predictions, subject metrics and paired differences under
`artifacts/faced_scale_v22/external_seed_vii/`. Execution completed successfully;
all target window identities are unique and all four model hashes matched.
