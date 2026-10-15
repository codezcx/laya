# Emotion frequency/context factorial v20

Fixed2x2 comparison: bandpass1–30 versus1–45Hz, and contiguous3 versus9seconds.
Use the same30 channels, source subjects, trials, labels and window anchors as
v16/v19. Existing3second windows remain exactly fixed. Each9second window is
centered around the corresponding3second window and shifted only when necessary
to remain within its trial; it always contains the original3seconds. Save exact
bounds. Do not pad with another trial or label, and do not call9second results
3second inference. Some neighboring long windows may overlap within a trial;
all windows of a subject remain in one split, and intervals resample subjects.

Read amplitude-preserving MAT/BDF signals anew, filter complete trials, no
reference change or per-channel clipping. Identical Welch1second segments with
50% overlap; longer windows provide more segments, not higher FFT resolution.
Bands1–4,4–8,8–13,13–20,20–30 plus30–45 in the wider setting. Normalize powers
within each channel to log fractions across included bands. Therefore the wider
comparison changes both upper bandpass cutoff and fraction denominator; it
is a combined wider-band representation test, not isolated gamma causation.
Distributed SEED files are already preprocessed; this cannot recover previously
removed frequencies. High-band energy is not automatically neural emotion
information and may contain artifacts.

Four CPU extraction shards, then four source-only fitting jobs (each both
directions). Full-channel linear logistic classifier only, the same C grid,
class weights and source validation selection as v16. Freeze all8 models before
transfer evaluation. No target fitting or window/band selection. Report all4
settings and both directions, shuffled controls and subject confidence
intervals. Re-extracted30Hz/3s must reproduce v19 as_recorded; discrepancies
require investigation. No target2b/SEED-VII access, no new task expansion.

Tests:9second bounds remain within trial and contain original3seconds; a
synthetic35Hz component appears in the added band. These support preprocessing
correctness, not empirical zero-shot success. Previously examined development
datasets remain exploratory, not fresh blind benchmarks.
