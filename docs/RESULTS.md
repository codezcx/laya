# LEAF baseline and JEV adaptation

Active: frozen released LEAF on2a four-class andSEED three-class, followed byJEV joint training on the same subject-disjoint adapter splits. Baseline output: `artifacts/leaf_2a_seed/baseline.json`. Preparation progress: `artifacts/leaf_2a_seed/prepare_progress.json`. JEV metrics: `artifacts/leaf_2a_seed/jev/metrics.jsonl`; controller status:`artifacts/leaf_2a_seed/pipeline.json`.

Native baseline completed (task-only instruction, equivalent to upstream level1):

| Dataset/metric | Train | Validation |
| --- | ---: | ---: |
| 2a four-class accuracy/BA | 36.28% | 37.15% |
| SEED three-class window accuracy | 64.72% | 65.00% |
| SEED trial accuracy | 71.60% | 67.41% |

2a validation predictions heavily favor feet; these are not official benchmark reproduction scores. Upstream also offers candidate-mean instruction blending (level2); this run keeps task-only conditioning fixed for native/JEV comparison. Features: train26,190 (2a3,456/SEED22,734),validation1,656(576/1,080). Source finalization initially repeated hashes per window; extraction finished, and finalization recovered from intact arrays after changing to50unique source hashes. Shapes,finiteness,unique IDs and subject disjointness passed. Nine tests passed. No JEV quality result yet. Existing two-MAT-window smoke was an interface check only. Historical REVE experiment results and code are archived in Git throughc75a101; all ignored artifacts retained. See [PROTOCOL.md](PROTOCOL.md).
