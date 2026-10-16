# Emotion spatial amplitude information v26

MI experimentsv23/v24 motivate testing spatial amplitude ratios in emotion,
without assuming effects transfer across tasks. Existing emotion log fractions
normalize each electrode separately and discard its total1–30Hz power relative
to other electrodes. Test this information loss with equal-dimensional features.

Use the exact32,827 source development windows fromv22:5,400SEED and27,427FACED,
including the existing13-window quality exclusion. Reread amplitude-preserving
MAT/BDF trials through thev19 raw-signal loader: same common30 channels,200Hz,
1–30Hz full-trial filtering,3s windows, original reference. No per-channel
zscore or clipping before spectral extraction. No reserved test or target load.

Welch configuration and five bands remain unchanged. Both arms output30×5
log powers. The control subtracts logsumexp across bands within each electrode;
joint_power subtracts logsumexp across all electrodes and bands. Both remove
global amplitude scale, avoiding a simple volts-versus-MAT-units distinction;
only joint_power retains inter-electrode relative total power. Channel-specific
gain/reference differences remain a potential domain confound. Neither feature
requires subject-wide or target-wide normalization.

Four CPU workers reread source recordings. Join only after exact index coverage,
manifest hashes, finite feature and split checks. Compare control againstv22's
cached fractions to quantify numerical differences from normalized/clipped input.
Training plan: source-only StandardScaler+class/dataset-weighted logistic,
C grid[.001,.01,.1,1,10]. For each arm, train on SEED train and validate FACED
validation, then reverse; choose by equal mean held-dataset subject-balanced
accuracy, ties smallerC. Refit on both training sets. Require convergence before
freezing both models. These are reused development domains, not unbiased tests.

External SEED-VII HNS evaluation is separate, only after source freezing; both
arms must be reported without target selection. Existing target exposure makes
it exploratory. This signal diagnostic does not establish text alignment or
task recognition; MI baselines remain unchanged and no new tasks are introduced.
