# Cross-source supervised signal adaptation v21

Move from frozen readout/preprocessing changes to representation learning.
Use SEED and FACED training subjects jointly, fixed30-channel3second1–30Hz
inputs and the same source-validation subjects as preceding protocols.
No target data participate in fitting or model selection.

Cache frozen REVE output immediately before its final two Transformer blocks.
Check fp32 tail replay against the original full encoder for every extraction
batch (maximum absolute error<=1e-5); save fp16 prefixes with row indices,
manifest hash and exact channel order. Unlike v17, the adapted arms update the
last two encoder blocks, not only a decision head.

Four equal-schedule arms: frozen_ce, tail_ce, tail_align, tail_align_spectral.
All share LayerNorm/Linear15360→256/GELU/dropout0.1 and a3-class head; auxiliary
projection128 and spectral reconstruction150 are instantiated in every arm.
Seed42,1000steps, batch48 (eight examples from each source/class group), AdamW
headlr1e-4, taillr1e-5, decay0.01, gradientclip1. CE is always active. Alignment
weight0.1 and spectral MSE weight0.1 start after100steps and ramp to full over
100 more steps. Select checkpoint by mean source-dataset subject-balanced
validation accuracy every100steps. Same-class cross-source pairs are positives;
only other-source examples enter the contrastive denominator (temperature0.1).
This is supervised representation learning from class labels, not report-level
EEG/text alignment. Fixed class names are not treated as independent evidence.

Spectral targets are v16's150 log-band-fraction features standardized using
source training rows only. The spectrum head is training-only; inference needs
signal alone. Do not claim downstream output improves merely because the
auxiliary loss falls. Frozen_ce and tail_ce isolate encoder tuning from added
losses. This changes the emotion branch only; MI results remain unchanged.

After all four source checkpoints freeze, evaluate all of them against the same
held target with no target checkpoint/seed choice. Target datasets have been
used in prior development, so any such evaluation remains exploratory. Source
training completion alone is not zero-shot success. Keep v16 frequency baseline
as a reference; do not add stress/arithmetic until the original objective is
met. The full model still lacks a validated universal task-identification path.
