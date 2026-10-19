# Current MI experiment status

Current approach: full-channel, full-exported-duration left/right MI classification with frozen REVE and trainable JEV LoRA/connector/A-B head. Protocol: [PROTOCOL.md](PROTOCOL.md).

- Train:49,273 trials from9source datasets; validation:12,508 subject-disjoint trials.
- Test:6,520 trials from held-out2b. No test results yet.
- Outputs:`artifacts/mi_jev/`; live phase:`status.json`; train/validation:`run/metrics.jsonl`; best checkpoint:`run/best.json`; final test:`run/test_2b.json`.
- Full-channel token preparation and four-GPU smoke must pass before training. This file does not claim completed training or improved accuracy.

TUAB was stopped at user request at step265/725. Best source validation accuracy78.41% (step150). Its logs/checkpoints remain at`artifacts/tuab_jev`; stopped state and original implementation are archived in Git commit`7335cfd`. MI uses original JEV initialization, not TUAB trained weights.
