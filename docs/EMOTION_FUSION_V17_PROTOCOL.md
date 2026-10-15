# Matched-montage REVE / spectral fusion v17

Follow-up to v16's full-channel spectral transfer signal. This known-task
emotion diagnostic tests whether frozen REVE contributes complementary signal
beyond frequency features, and whether text prototypes improve the same fusion
input. It does not establish a universal model or automatic task recognition.

Use exactly v16's rows, subject splits and ordered 30 common channels. Re-encode
both SEED and FACED from those 30 channels before REVE contextual processing,
with identical canonical coordinates, rather than pooling differently sized
montages from old caches. Frozen REVE fp32 outputs are time-averaged to
30x512 and stored fp16. Existing v16 log band fractions provide 30x5 spectral
features. Window length, preprocessing and label definitions remain unchanged.

Four arms, each trained in both directions:
- spectral: 150-dimensional frequency input;
- reve: 15,360-dimensional frozen encoder input;
- fusion: concatenate those inputs;
- fusion_text: same concatenation, output normalized similarity to three fixed
  centered Qwen emotion text prototypes instead of an ordinary linear head.

Each arm uses a 256-wide GELU/dropout0.1 hidden layer, CE, AdamW lr1e-4,
weight_decay0.01, seed42, 3,000 steps, batch96 and class-balanced sampling.
Feature mean and variance are computed only over source training subjects and
saved with the checkpoint; std is floored at0.001. Source validation mean
subject balanced accuracy selects every250steps. Parameter counts differ with
input dimension and text output, and are recorded; this is not a parameter-
matched comparison. The v16 regularized linear spectral baseline remains the
reference, not just the new nonlinear spectral head.

All eight checkpoints must complete and their hashes freeze before evaluating
held datasets. No target fitting, checkpoint selection, or target 2b/SEED-VII
access. Report both transfer directions, ordinary and balanced accuracy,
subject-bootstrap intervals and fixed prediction shuffle. These datasets were
used in prior development; call results exploratory dataset transfer, not a
new blind test. No arithmetic/stress expansion follows merely from a better
emotion score.
