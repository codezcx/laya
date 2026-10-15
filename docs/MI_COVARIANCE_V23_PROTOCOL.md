# MI relative-amplitude diagnostic v23

The complete three-source MI pool has not established strong cross-dataset
transfer. Before further language-interface changes, test whether removing
inter-electrode variance is a meaningful information bottleneck. Older
harmonized-input experiments exist; this is a matched diagnostic on the audited
37,432-trial full source pool, not a claim that shared normalization is new.

Use C3/Cz/C4 and first3seconds at200Hz. Filter complete available trials before
cropping, using fixed fourth-order bands8–12,12–16,16–20,20–24,24–30Hz. Each band
produces a3×3 covariance. Compare correlation normalization against covariance
divided by its trace and multiplied by3. Both use fixed1% identity shrinkage and
matrix-log upper triangles, with sqrt(2) off-diagonal weights:30 features each.
Both are invariant to common multiplicative gain; only relative covariance
retains per-electrode variance ratios. Differences in noise/reference may still
make those ratios unhelpful. No subject or target aggregate normalization.

Source data: unchanged Cho2017, Dreyer2023, Lee2019_MI train/validation manifest
and audited keep indices fromv12. Original source tests remain excluded. Four
CPU extraction workers read the same raw trials for both representations.

Selection plan fixed before fitting: for each representation and each
C in[.001,.01,.1,1,10], hold one entire dataset out of training. Fit standardizer
and logistic regression on the other datasets' training subjects only, with
equal dataset/class sample weights. Score the held dataset's existing validation
subjects. Select C by mean subject-macro accuracy across the three held-domain
folds, ties preferring smallerC. Refit on all three training subsets; report
within-source validation separately. Freeze both final models before any 2b
evaluation. The source datasets have prior development history, so these are
development folds, not independent blind evidence.

This is a signal representation diagnostic, not an LLM or semantic-alignment
model. No target labels, new tasks, or target adaptation enter this stage. A
positive result would motivate a controlled REVE input change; a negative result
would prevent another unsupported interface change. Neither result by itself
satisfies the universal zero-shot objective.
