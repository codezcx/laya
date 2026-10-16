# Continuous subjective-rating auxiliary control v28

Test whether FACED's supplied subjective Valence adds useful supervision beyond
stimulus classes. Treat the TSV values as reported scores, not clinical ground
truth or recoded class labels. No user/subject exclusion or rating thresholds.

Usev26 raw-float64 per-channel fractions (150features) and its exact source
manifest. Hold one FACED video per fine emotion category, sampled from sorted
IDs with seed2801:9videos absent from all fitting. Preserve73/25FACED subject and
existing SEED subject splits. Training uses only source train subjects and
non-held FACED videos; validation reports unseen-subject/unseen-video FACED,
unseen-subject/seen-video FACED, and SEED validation separately. SEED trial IDs
are not asserted to represent identical clips across sessions. Source tests and
external targets remain excluded. This is source development, not a new test set.

Four arms:classification only; actual Valence; Valence permuted among unique
training trials within stimulus class (seed2802); training class-mean Valence.
All windows from a trial share the same auxiliary target. Auxiliary rating
mean/std and class means are computed only from unique retained training trials.
No validation ratings enter fitting or selection. Within-class shuffling retains
class-level information while destroying individual trial association; the
class-mean arm controls for extra label-based regularization.

All arms instantiate identical150→128→64 GELU trunks,dropout.1,3-class and
scalar regression heads,seed42. Use AdamW lr1e-3,weight_decay.01,gradient clip1,
2000steps,batch192 (32samples per dataset/class group). Identical sampling and
initial weights across arms. Loss=class CE+.2*SmoothL1 on standardized FACED
ratings; classification arm uses zero auxiliary weight. All feature statistics
fit only training rows. Fixed protocol, no hyperparameter sweep.

Evaluate every100steps, selecting the mean of SEED and unseen-video FACED
subject-balanced accuracy; exact ties prefer earliest checkpoint. Report all
four arms; only consider later external evaluation after source hashes freeze.
No Qwen or REVE forward is used here: this is a supervision diagnostic before
spending further effort on a language connector. It cannot establish semantic
alignment or automatic task recognition. MI unchanged; no new tasks enabled.
