# Frozen probability fusion v18

Follow v17's failure of feature concatenation by preserving the spectral model
as a direct decision path. Reuse v16 full_linear and v17 reve frozen models,
verify completion hashes and matching class order. Two source directions:
SEED→FACED and FACED→SEED. Same subjects/windows/common30 montage as before.

Blend probabilities as (1-alpha)*spectral + alpha*REVE. Choose alpha from
[0,.25,.5,.75,1] using only source validation subject balanced accuracy. Ties
choose smaller REVE weight. Persist both source choices and model hashes before
held predictions. No target fitting, normalization or alpha choice. Report
selected fusion and its two component baselines, along with paired-subject
bootstrap differences (10,000 draws, seed2026). There is no extra training and
one GPU suffices for frozen REVE-head inference; raw EEG is not re-encoded.

This tests complementary decisions, not semantic text alignment. The source
validation subjects already selected component checkpoints, and both datasets
were used in earlier development: this extra selection can overfit validation
and is not a fresh blind test. Do not claim robust improvement from source
validation alone. No 2b/SEED-VII access, no arithmetic/stress task expansion.
