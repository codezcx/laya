# Joint cross-subject benchmark status

Current: online LEAF→JEV joint fold0 on11classification datasets, 100-step frozen warmup followed by encoder fine-tuning. Previous frozen-query preparation was stopped and its ignored artifacts retained. No new SFT accuracy has been established yet. See [protocol](PROTOCOL.md). Status: `artifacts/benchmark/cross_subject_fold0/pipeline.json`; metrics: `artifacts/benchmark/cross_subject_fold0/jev/metrics.jsonl`. This is a single joint cross-subject fold, not complete per-dataset cross-validation.

The two-step four-GPU smoke run passed on all11datasets and both option orders. Encoder gradient norm was0during frozen warmup and4.4612after unfreezing; the tokenizer parameter-update assertions passed. These two-step predictions are functional checks, not benchmark accuracy. Formal3,000-step SFT has been launched.

Fold0 contains476,032training epochs,58,362validation epochs and60,275test epochs. Full split/count audit: `datasets/benchmark/signals/fold0/complete.json`.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| BNCI2014001 | 2016 | 288 | 288 |
| BNCI2014004 | 1120 | 120 | 160 |
| BNCI2015001 | 2000 | 200 | 200 |
| BNCI2014008 | 25200 | 4200 | 4200 |
| BNCI2014009 | 4608 | 576 | 576 |
| CHB_MIT | 23994 | 491 | 1807 |
| TUAB | 43218 | 5148 | 5508 |
| SleepEDF | 327294 | 43735 | 43932 |
| SEED | 44122 | 3394 | 3394 |
| EEGMAT | 1020 | 30 | 30 |
| Nakanishi2015 | 1440 | 180 | 180 |

Historical completed LEAF results below use different inputs/splits and must not be directly substituted for this benchmark.

# LEAF baseline and JEV adaptation

Completed matched control: frozen released LEAF on2a four-class andSEED three-class, followed byJEV joint training on the same subject-disjoint adapter splits. Baseline output: `artifacts/leaf_2a_seed/baseline.json`. Preparation progress: `artifacts/leaf_2a_seed/prepare_progress.json`. JEV metrics: `artifacts/leaf_2a_seed/jev/metrics.jsonl`; controller status:`artifacts/leaf_2a_seed/pipeline.json`.

Native baseline completed (task-only instruction, equivalent to upstream level1):

| Dataset/metric | Train | Validation |
| --- | ---: | ---: |
| 2a four-class accuracy/BA | 36.28% | 37.15% |
| SEED three-class window accuracy | 64.72% | 65.00% |
| SEED trial accuracy | 71.60% | 67.41% |

2a validation predictions heavily favor feet; these are not official benchmark reproduction scores. Upstream also offers candidate-mean instruction blending (level2); this run keeps task-only conditioning fixed for native/JEV comparison. Features: train26,190 (2a3,456/SEED22,734),validation1,656(576/1,080). Source finalization initially repeated hashes per window; extraction finished, and finalization recovered from intact arrays after changing to50unique source hashes. Shapes,finiteness,unique IDs and subject disjointness passed. Nine tests passed. JEV launched on GPUs2–3 after successful two-rank smoke (forward/backward plus normal/reversed validation). Initial smoke failed only its unsorted stratified-index equality assertion; indices sorted, failed artifacts preserved with .failed_order_check suffix, rerun passed. Production reached step2/600, finite gradients/loss, peak allocated54.07GiB; no full JEV validation yet. Step1 checkpoint saved. Existing two-MAT-window smoke was an interface check only. Historical REVE experiment results and code are archived in Git throughc75a101; all ignored artifacts retained. See [PROTOCOL.md](PROTOCOL.md).


Four-GPU restart: two-GPU run stopped at logged step77 by user request, before full step100validation. Artifacts preserved in stopped_2gpu_step77/. Restart fresh on GPUs0–3 with global batch48 and per-GPU microbatch6;600steps. No old training state reused.

Four-rank smoke passed, then fresh production reached step8/600. Steady early step times approximately5.4seconds versus8.1seconds on two GPUs (~1.5x throughput); validation time not yet measured. Peak allocated54.07GiB per rank0; original global batch48 preserved.


## Completed four-GPU JEV result

Finished600steps in3,403.26seconds(56.72minutes). Final step600 is also validation-selected best. Normal-order validation:2a54.3403%(+17.1875points vs native LEAF37.1528%),SEED window68.5185%(+3.5185points),SEED trial77.0370%(+9.6296points). Reversed order:2a54.5139%,SEED window68.2407%,SEED trial74.8148%. Window semantic prediction consistency95.8333%/98.6111% for2a/SEED. No test evaluation; released LEAF training exposure remains unresolved, not zero-shot evidence.

Important class failure:2a normal-order tongue recall3/144=2.0833%;135/144tongue trials predicted feet. Other recalls:left84.03%,right57.64%,feet73.61%. Thus overall improvement does not establish reliable four-class decoding.2a validation contains one held-out adapter subject(A07), limiting generalization claims. Files:artifacts/leaf_2a_seed/jev/validation_0600.json,best.pt,last.pt. Last-batch training accuracy72.92% is not full-training-set accuracy.


## Frozen transfer inference on other MI datasets

Completed5,193trials in176.49seconds; noSFT,optimizer,threshold fitting or weight updates. Same step600best.pt; candidates changed to left/right. Fixed previous validation subsets,not complete source datasets.

| Dataset | Trials / subjects | Native LEAF accuracy | JEV normal accuracy | JEV reversed accuracy | Semantic consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cho2017 | 2000 / 10 | 60.30% | 61.35% | 62.30% | 97.35% |
| OpenBMI (Lee2019_MI) | 2200 / 11 | 66.73% | 75.82% | 73.95% | 96.05% |
| PhysionetMI | 993 / 22 | 66.16% | 69.99% | 70.19% | 96.78% |

Improvements over native normal-order baseline:+1.05,+9.09,+3.83percentage points. Normal JEV still left-biased:Cho left/right recalls86.50%/36.20%;Physionet97.18%/42.86%. OpenBMI81.73%/69.91%. Infer transfer performance under recorded preprocessing,not proof of universal robustness. Results,subject breakdowns,confusion matrices and source/checkpoint provenance:artifacts/leaf_2a_seed/external_mi/results.json,config.json,predictions.jsonl. Upstream pretraining membership was not a selection criterion per user request.
