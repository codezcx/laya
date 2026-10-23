# Frozen MI+SEED encoder probe diagnostic

Current job extracts frozen step1538 REVE features from `contrast_mi_seed/last.pt` and fits independent512-dimensional linear MI/SEED classifiers after discarding the contrast projection. It does not yet train separate contrastive projection heads. Output: `artifacts/mi_emotion/probe_mi_seed_encoder/`; final report `results.json`. Train/validation splits unchanged.

# Active MI + SEED three-class matched control

Only MI and SEED valence supervise the current contrast run; SEED-VII loss is disabled. Weights0.5/0.25, original REVE,1,538steps and complete49,193-trial MI coverage. Metrics: `artifacts/mi_emotion/contrast_mi_seed/metrics.jsonl`. Completed1,538steps in1,354.59seconds. Final MI macro dataset BA55.87%, SEED trial accuracy55.56%. Matched final MI BA references: MI-only81.90%, all-three-task54.38%. Thus adding only SEED supervision also reduces MI performance under this recipe (26.03 percentage points relative to MI-only); the seven-class task is not necessary for the failure. This does not establish that all multitask architectures are incompatible. The inherited three-task selector chose step1200 (MI BA53.63%, SEED accuracy59.26%); final-step comparisons use validation_1538.json/last.pt. No JEV continuation was started.

# Active MI-loss-only matched control

Current run: `artifacts/mi_emotion/contrast_mi_loss_only/metrics.jsonl`. Disable only emotion losses; MI weight remains0.5, full49,193-trial epoch, original REVE and identical initialization/text/sampling/schedule. The full joint baseline remains `contrast_full_mi`. Completed1,538steps. Final MI macro dataset BA81.90%, versus54.38% in the matched joint run (+27.52 percentage points). MI trajectory: step40066.22%,60075.76%,80076.65%,100080.07%,120080.46%,140080.37%,153881.90%. Final SEED trial accuracy36.30%, SEED-VII12.81% (emotion supervision disabled). This matched seed/run supports a detrimental effect of emotion supervision under the current shared model/objective; it does not isolate gradient magnitude as the sole mechanism. The inherited aggregate selector chose step1400 for best.pt; final step1538 is in last.pt and has the highest observed MI BA. No JEV continuation.

# Current contrast-only rerun

Prior joint-training jobs were stopped by the user. Current experiment: one full shuffled MI epoch,49,193 unique MI trials in1,538steps, plus SEED/SEED-VII sampling. Initialized from original REVE, not the failed joint encoder or prior MI-only encoder. No automatic JEV continuation. Metrics: `artifacts/mi_emotion/contrast_full_mi/metrics.jsonl`; coverage audit: `artifacts/mi_emotion/full_mi_coverage.json`. Completed1,538steps in1,444.81seconds; best checkpoint is the final step1538. MI macro dataset BA54.38%, SEED trial accuracy62.22%, SEED-VII trial accuracy25.63% (BA26.04%). No JEV continuation was started.

Previous joint contrast best (step800): MI macro dataset BA53.12%, SEED trial accuracy57.78%, SEED-VII trial accuracy26.25%. Previous MI-only contrast macro dataset BA76.53%. These distinguish stage1 quality from downstream JEV performance. Historical artifact directories are preserved.

# Previous joint MI/emotion run

Implementation and protocol: [PROTOCOL.md](PROTOCOL.md). GPU2–3 only. Preparation completed: MI49,193 trials, SEED22,734 windows/405 parent trials, SEED-VII41,844 windows/960 parent trials in training. Validation is fixed and subject-disjoint within each dataset. Both two-GPU smoke checks passed (contrast backward/validation and JEV mixed-choice backward/paired validation); 12 unit/distributed tests passed. Contrast completed800steps. The first JEV run was interrupted by node loss at step173, before any checkpoint was saved. A fresh JEV run using the selected contrast encoder is prepared under `jev_restart`; training microbatch4, validation microbatch8, length-sorted validation. Smoke metrics are execution checks, not model-quality results.

- Pipeline: `artifacts/mi_emotion/pipeline.json`
- Contrast: `artifacts/mi_emotion/contrast/metrics.jsonl`
- JEV: `artifacts/mi_emotion/jev_restart/metrics.jsonl`
- Best validation and checkpoint: each stage's `best.json` and `best.pt`
- Data inventory/splits: `datasets/mi_emotion/complete.json`

Previous MI-only experiments and results remain in Git and their original ignored artifact directories. They are not joint-task results. The prior two arms stopped at logged step1200; saved validation ended at1000. At step600, normal-option source validation accuracy was74.55% for CE preconditioning and77.19% for text contrast preconditioning. These are subject-disjoint source-validation accuracies, not training or unseen-dataset test results.

Node-restart checks passed: nine related tests; two-rank longest-input forward/backward with training microbatch4 and paired validation microbatch8. Rank0 peak allocated52.85GiB/reserved56.96GiB across the complete check. Recovery checkpoint saved before validation. Production restart uses GPUs2–3, output `artifacts/mi_emotion/jev_restart/`.

## Diagnosis after full MI coverage

Full coverage did not restore the previous MI-only macro BA76.53%. Final MI dataset accuracies:2a60.24%, Chang54.79%, Cho54.30%, Dreyer54.98%, Kumar52.31%, OpenBMI57.23%, Physionet52.97%, Weibo51.00%, Zhou51.41%.

The last100 full batches have mean MI loss3.46497, essentially the uniform-score baseline ln(32)=3.46574. Thus the training objective itself is barely improved, not just held-out classification. SEED/SEED-VII losses fall to2.04243/2.22725 from the uniform ln(16)=2.77259 baseline. Loss values cannot be compared directly across different task batch sizes.

Saved text prototypes: MI left/right cosine0.98509 (old successful MI-only experiment0.98684); SEED distinct-class average0.91031; SEED-VII0.87435. High MI text similarity may make its signal weaker, but it existed in the successful control and cannot alone explain failure.

Read-only gradient probe: `scripts/diagnose_joint_gradients.py`, final checkpoint, three fixed train batches (sampling steps200/800/1200), train mode, complete encoder plus shared projection gradients, no optimizer updates. Results in `artifacts/mi_emotion/diagnosis/gradients.json`; compute norm/cosine in float64 to avoid long-vector float32 reduction error. Weighted MI gradient norm is about0.4–0.5, while each emotion task is approximately1.8–8.6. This supports investigating task-gradient scale imbalance despite nominal50/25/25 loss weights. MI/emotion cosine values are near zero or mildly positive/negative, not consistently strongly opposing. These are final-checkpoint diagnostics on three batches, not causal evidence or a direct measurement of AdamW parameter-update shares.

Next discriminating controls: run MI-only with the current text/sampler/implementation to isolate regressions; if it recovers, vary only task-loss scaling in the joint run and monitor per-task gradients/validation. Do not start another long JEV stage on the current encoder merely because its aggregate score improved.
