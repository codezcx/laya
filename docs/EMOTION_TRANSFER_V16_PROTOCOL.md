# Emotion spectral transfer diagnostic v16

Purpose: test whether common-montage frequency features transfer between SEED
and FACED before another encoder/alignment change. This is a known-emotion-task
diagnostic, not the final universal model. No 2b or SEED-VII evaluation.

Use the existing fixed train/validation subjects and windows. Fit on one
emotion dataset's training subjects, select C only on its validation subjects,
then evaluate on all development subjects of the other dataset. All eight
selected model hashes must be frozen before either transfer direction is
reported. These datasets have been examined previously; this is exploratory
cross-dataset development, not a fresh blind benchmark.

Use 30 common scalp channels, excluding references/auxiliary EOG; map legacy
T3/T4/T5/T6 to T7/T8/P7/P8 explicitly. Frequency bands: 1–4, 4–8, 8–13,
13–20, 20–30 Hz. Features are log band-power fractions normalized within each
channel/window; no target mean, variance, labels, or optimization is used.
SEED reuses pre-clip cached band powers; FACED uses prepared zscore/clip15
waveforms. Fraction normalization cancels per-channel variance scaling, but
rare clipping can still differ. Both use 200 Hz, Welch nperseg200/noverlap100.

Four predeclared comparisons, in each direction:
1. C3/Cz/C4 log fractions, linear logistic regression.
2. All 30 channels, linear logistic regression.
3. All 30 channels with per-window spatial mean removed separately per band,
   linear logistic regression. This tests suppressing global spectral offsets.
4. The same spatially centered features, RBF SVM (gamma=scale).

StandardScaler is fitted only to source training rows. Balanced class sample
weights account for FACED's 12/4/12 trial distribution. C grid is fixed at
0.001, 0.01, 0.1, 1, 10. Select source validation mean subject balanced accuracy;
report ordinary accuracy too, convergence warnings, a fixed prediction shuffle,
and 10,000-draw subject bootstrap intervals. No held-dataset selection. This
small classical baseline runs as four CPU jobs; GPUs are unnecessary for it.

Training/evaluation use separate stages, fail on existing output directories,
and preserve artifacts. A useful result must improve actual class transfer,
not task routing. If common-channel frequency transfer is also at chance,
repeatedly replacing the output text is unlikely to resolve the observed gap;
full-channel signal representation and acquisition/label compatibility remain
open hypotheses, not established causes.
