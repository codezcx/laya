# v31: Fixed source-only training-budget check

## Why this round

Both v30 arms selected the last candidate (step 500), and source validation was still improving near the boundary. Those 500 updates sampled only 24,000 examples with replacement across a 57,749-row training set. This does not prove undertraining, but it leaves convergence unresolved. Test a larger fixed budget before changing representations, prompts or encoder weights. This decision is motivated by the source curves; the v30 targets have already been exposed and remain exploratory.

## Fixed recipe

Run both Qwen and parameter-matched signal arms for **3,000 updates**, sequentially on four GPUs. Use the exact unchanged v30 trainer and verified full-token source cache. Initialize afresh with the same seed and original Qwen checkpoint; do not reload connector-only checkpoints as if they included optimizer state. Sampling, optimizer/lr, batch size, prompt, freeze policy and source selection metric remain unchanged. Checkpoints are evaluated at steps 0,100,…,3000; select by the same task/dataset/subject-balanced joint accuracy, earlier candidate wins ties. This increases the number of source selection opportunities and is not a new unbiased validation.

The nominal sampled training volume is 144,000 examples (with replacement, task/source/class balanced), not a guarantee of visiting every source row. Compare the first 500-step source trajectory against v30 as a reproducibility check, and report the selected step and full source curves. Distinguish source improvement from external transfer; even converged models may fail both targets.

## Execution and evidence

`scripts/run_bridge_budget_v31.py` checks the v30 trainer source hash and cache audit before launching. It runs Qwen then signal, never restarts a failed job, and records process IDs and status under `artifacts/multitask_bridge_v31`. Each arm has its own output folder and log. Existing v30 artifacts remain intact. The controller does **not** load targets or launch external evaluation. Only after both source fits finish and their selected model hashes are frozen may a later evaluation use the existing identical external inputs.

Expected runtime from v30 is roughly 25 minutes for Qwen plus about one minute for the control, subject to actual throughput. Training remains restricted to MI and emotion; no arithmetic/stress expansion is justified by v30.

## Prefix reproducibility audit

`scripts/audit_bridge_budget_prefix_v31.py` compares run metadata, all logged losses through step 500 and source validation records at steps 0–500 with v30. It ignores wall-clock timings, requires the budget to be the only metadata difference, reports numerical differences rather than silently accepting them, and marks the audit incomplete until all expected prefix records exist. A live writer's unfinished final line is not parsed as a complete record. Results are written atomically to `artifacts/multitask_bridge_v31/prefix_audit.json`.

Initial inspection at step 120: metadata differ only in steps [500,3000], and available loss records plus validation at steps 0 and 100 reproduce v30 exactly (maximum numerical differences zero). This is a partial prefix check; the full 500-step prefix and signal arm remain pending. The existing controller and training process were confirmed live; no retry or restart occurred.
