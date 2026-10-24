# LEAF baseline and JEV adaptation

Active: frozen released LEAF on2a four-class andSEED three-class, followed byJEV joint training on the same subject-disjoint adapter splits. Baseline output: `artifacts/leaf_2a_seed/baseline.json`. Preparation progress: `artifacts/leaf_2a_seed/prepare_progress.json`. JEV metrics: `artifacts/leaf_2a_seed/jev/metrics.jsonl`; controller status:`artifacts/leaf_2a_seed/pipeline.json`.

Native baseline completed (task-only instruction, equivalent to upstream level1):

| Dataset/metric | Train | Validation |
| --- | ---: | ---: |
| 2a four-class accuracy/BA | 36.28% | 37.15% |
| SEED three-class window accuracy | 64.72% | 65.00% |
| SEED trial accuracy | 71.60% | 67.41% |

2a validation predictions heavily favor feet; these are not official benchmark reproduction scores. Upstream also offers candidate-mean instruction blending (level2); this run keeps task-only conditioning fixed for native/JEV comparison. Features: train26,190 (2a3,456/SEED22,734),validation1,656(576/1,080). Source finalization initially repeated hashes per window; extraction finished, and finalization recovered from intact arrays after changing to50unique source hashes. Shapes,finiteness,unique IDs and subject disjointness passed. Nine tests passed. JEV launched on GPUs2–3 after successful two-rank smoke (forward/backward plus normal/reversed validation). Initial smoke failed only its unsorted stratified-index equality assertion; indices sorted, failed artifacts preserved with .failed_order_check suffix, rerun passed. Production reached step2/600, finite gradients/loss, peak allocated54.07GiB; no full JEV validation yet. Step1 checkpoint saved. Existing two-MAT-window smoke was an interface check only. Historical REVE experiment results and code are archived in Git throughc75a101; all ignored artifacts retained. See [PROTOCOL.md](PROTOCOL.md).


Four-GPU restart: two-GPU run stopped at logged step77 by user request, before full step100validation. Artifacts preserved in stopped_2gpu_step77/. Restart fresh on GPUs0–3 with global batch48 and per-GPU microbatch6;600steps. No old training state reused.

Four-rank smoke passed, then fresh production reached step8/600. Steady early step times approximately5.4seconds versus8.1seconds on two GPUs (~1.5x throughput); validation time not yet measured. Peak allocated54.07GiB per rank0; original global batch48 preserved.
