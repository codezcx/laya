# MI complementary-information test v25

v23 relative covariance outperformed frozen REVE on2b; v24 common scaling
improved REVE over per-channel scaling. Test whether the two representations
have complementary source-generalizable decisions before further encoder or
language-interface training.

Use unchanged source indices and labels, verified by both feature-cache audits
and exact manifest-row equality. Reuse v23 relative-covariance C=.001 and v24
converged common-scale REVE C=.01, chosen on source folds. For each representation,
refit on two datasets' training subjects and predict only the third dataset's
validation subjects. Fit standardizers and weights only on each training fold.
Require all six models to converge with max_iter10,000.

Mix probabilities as(1−alpha)*covariance+alpha*REVE for fixed
alpha[0,.25,.5,.75,1]. Select by equal mean dataset subject-macro accuracy over
the three held-domain prediction sets, preferring smalleralpha on exact ties.
This is source development selection, not nested unbiased cross-validation:
the C values and mixing weight share development validation history. Do not
claim these scores are independent test estimates.

Freeze selected alpha and both existing full-source model hashes before target
evaluation. Report covariance, REVE and the source-selected mixture on exactly
the same2b E trials; do not choose an alternate mixture using target scores.
Target evaluation is exploratory given prior exposure, and REVE pretraining
includes2b. If alpha=0, retain the baseline without pretending fusion helped.
This remains a known-task signal classifier, not semantic task identification.
Emotion unchanged; no task expansion until the full objective has evidence.
