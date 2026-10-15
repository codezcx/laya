# Frozen probability fusion v18 results

Both source-validation-selected probability mixtures completed. Model hashes and both alpha choices were persisted before held-dataset prediction. Two probability-validation/endpoint tests passed. See [protocol](EMOTION_LATE_FUSION_V18_PROTOCOL.md).

| Source → held dataset | Selected REVE weight | Spectral balanced % | REVE balanced % | Selected mixture balanced % | Paired change vs spectral, pp (95% CI) |
|---|---:|---:|---:|---:|---:|
| SEED → FACED | 0.75 | 38.16 | 36.86 | 36.97 | -1.19 (-2.91, 0.50) |
| FACED-NEMAR → SEED | 0.00 | 43.41 | 38.74 | 43.41 | 0.00 (0.00, 0.00) |

## Interpretation

Source-selected late fusion also fails to improve emotion transfer. FACED source validation chooses alpha0, retaining the spectral model unchanged. SEED source validation chooses alpha0.75, but held FACED falls from38.16% to36.97%. The source-selected mixtures are reported unchanged; target scores were not used to replace alpha afterward.

This closes the simple feature/probability-fusion branch for the current frozen REVE readout. Neither adding feature dimensions, a fixed text prototype head, nor mixing predictions has demonstrated transfer improvement over the frequency baseline. It does not prove that REVE cannot be useful with another objective or representation.

A more consequential next step is to audit reference/preprocessing compatibility and train representations against class-relevant frequency/spatial supervision, or use source-domain validation that reflects transfer rather than merely new subjects in the same acquisition pipeline. Repeating classifier-head variants on the same validation subjects is unlikely to resolve the observed source/transfer mismatch.

## Scope and artifacts

Known-task emotion classification only; not automatic task recognition or unseen-task zero-shot. No MI update or 2b/SEED-VII evaluation. Repeated development datasets and source-validation reuse limit inference; confidence intervals capture subject sampling only. Goal remains incomplete and no stress/arithmetic expansion follows.

`artifacts/emotion_late_fusion_v18/selection.json` contains every source-validation alpha score and frozen component hashes; `results.json` contains ordinary/balanced scores, subject intervals and paired differences; probability arrays preserve both components and the chosen mixture. `complete.json` hashes the selection record.
