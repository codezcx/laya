# v30: Real Qwen bridge and matched token readout

## Frozen-Qwen arm completed

The 500-update four-GPU run completed successfully. The source-only selection rule chose step 500 from steps 0/100/200/300/400/500. There were 57,749 training and 12,510 validation windows. Only the 18,883,584-parameter connector trained; REVE, Qwen and the original A–E vocabulary readout stayed frozen. The training/evaluation loop took 259.2 seconds, excluding initial model loading and cache extraction.

| Source validation dataset | Subject-macro balanced joint accuracy |
|---|---:|
| Cho2017 | 56.60% |
| Dreyer2023 | 62.50% |
| Lee2019_MI | 54.10% |
| SEED | 44.81% |
| FACED-NEMAR | 39.39% |

The task-macro selection score is 49.92%. Each dataset has 100% task routing, so oracle-task and joint class accuracy coincide here. This does not demonstrate task semantics: MI has 12 tokens from three channels while emotion has 120 tokens from thirty channels. Source subject generalization is not external dataset generalization.

Artifacts: `artifacts/multitask_bridge_v30/qwen_frozen/{run.json,validation.jsonl,best_metrics.json,best.pt,complete.json}`. The completion record hashes both best and last connector checkpoints. The entire source cache passed validation; independent real-smoke inputs also reproduced exact fp16 cached tokens when the encoder batch shape was matched.

## Signal control completed

`scripts/run_bridge_control_v30.py` was started while Qwen was running, verified the original controller was live, waited for successful completion, then verified the Qwen best checkpoint, trainer source and cache-audit hashes. It launched one four-GPU signal-control job with the same 500 updates, rank sampling seeds, optimizer and selection rule. The trainer source hash must remain identical between arms. No automatic retries occur on failures.

The signal model has 18,864,951 trainable parameters (within 0.1% of the Qwen connector), preserves channel/time positions and uses the same token cache. The control output directory is `artifacts/multitask_bridge_v30/signal_control`; progress is in `control_controller.json` and `control_training.log`. No control score is asserted until its completion record and selected checkpoint are checked.

The control completed all 500 updates and selected step 500. Best/last checkpoint hashes and matched protocol fields were verified for both arms. Control source balanced accuracy was Cho 56.60%, Dreyer 60.19%, Lee 55.80%, SEED 44.37%, FACED 36.79%; task routing was 100%. Its task-macro selection score is 49.05%, versus Qwen 49.92% (difference +0.86 percentage points). This small single-seed source difference is not evidence of a reliable language advantage. The control loop took 10.46 seconds versus Qwen's 259.22 seconds; these are logged training-plus-validation timings, not inference benchmarks.

`artifacts/multitask_bridge_v30/source_freeze.json` now records both source-selected checkpoints and their run/selection/completion metadata hashes before v30 external data extraction. Neither arm is promoted based on target performance.

## Interpretation and remaining work

This round verifies a real EEG→connector→Qwen training path, unlike offline text prototypes. It still trains from categorical labels, not paired descriptive EEG reports. No external 2b/SEED-VII evaluation has been run for v30. Language benefit, cross-dataset improvement and universal task recognition remain unproven. Next: evaluate both frozen source-selected checkpoints on identical external token inputs with signal-shuffle diagnostics. Both external datasets have already been used in exploratory development; REVE pretraining includes 2b.
