# Common-channel, task-free dataset transfer v15

The goal is to identify task and class without a supplied task ID. Earlier
masked losses and metrics did not train or verify this behavior. Different
channel counts also made task identification an unreliable semantic test.

This diagnostic uses the same C3/Cz/C4 input montage, 3-second windows, 200 Hz,
1–30 Hz preprocessing and frozen REVE across emotion and motor imagery.
FACED is re-encoded from those three channels before REVE, rather than selecting
channels after full-montage contextual encoding. Existing SEED and MI caches
were already encoded from this same three-channel montage.

Four runs: ordinary five-class head or text-prototype head, crossed with
held-out emotion dataset SEED or FACED. Lee2019_MI is held out in every run.
Training uses the other emotion dataset plus Cho2017 and Dreyer2023. Only
subject-held-out validation from training datasets selects checkpoints; no
held-out dataset contributes to fitting or selection. All available existing
train/validation records of held-out datasets are evaluated after all four
checkpoints are frozen. This is a development transfer test, not a fresh blind
test: these datasets have been used in previous experiments.

Targets: negative emotion, neutral emotion, positive emotion, left-hand motor
imagery, right-hand motor imagery. Both heads use unmasked five-class cross
entropy. Text prototypes are globally centered across the five classes (not
within tasks). Qwen supplies fixed offline text embeddings, not a decoder at
inference. Settings: seed42, 3,000 steps, batch96, AdamW lr1e-4, task/dataset/class
balanced sampling. Selection: source-validation macro-task, macro-dataset,
macro-subject class-balanced joint accuracy.

Report jointly predicted class accuracy, task accuracy from summed class
probabilities, and oracle-task class accuracy separately. Report a within-dataset
prediction shuffle: it preserves task/dataset information while destroying
class correspondence. Two unit tests verify wrong-task predictions cannot
receive masked correctness and whole datasets are excluded from fitting.

Common channels remove channel-count leakage, but do not eliminate differences
in hardware, protocol, filtering upstream, or participant populations. High task
accuracy alone is not evidence that the model understands EEG task semantics.
The three-channel constraint may hurt emotion recognition; this diagnostic
will not replace full-channel emotion experiments without supporting evidence.
No stress/arithmetic expansion is authorized by successful routing alone.
