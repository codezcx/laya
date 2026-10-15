# Matched frozen-REVE scaling comparison v24

v23's relative-covariance model transferred better to2b than its correlation
control. Test whether retaining spatial variance ratios helps frozen REVE too;
do not assume that bandwise classical features and broadband REVE respond alike.

Use all37,432 audited source train/validation trials fromv12, unchanged C3/Cz/C4,
200Hz, full-available-trial fourth-order1–30Hz filtering before first3s crop.
Both arms subtract each channel's time mean and clip normalized values to±15.
The control divides by each channel's standard deviation; the other divides
all channels by one trial RMS. No subject-wide or target-wide statistics.

Both arms use the same local REVE base checkpoint in
`configs/reve_qwen_mi_multi.json`, identical position embeddings, replicate
padding, frozen fp32 encoder and temporal pooling to3×512. Extract on four GPUs,
cache fp16 features with identical trial indices. Verify control input SHA256
againstv12 for every trial and audit replay differences against its frozen cache.
Stop for fp16 feature absolute differences above0.02; report actual maximum and
clipping counts rather than claiming exact replay from the tolerance alone.

Readout plan: separate standardizer+logistic models for each arm, same
C grid[.001,.01,.1,1,10], dataset/class weights, and three held-dataset validation
folds asv23. Select by equal mean across held datasets, smaller C on exact ties;
refit on all source training subjects. Original source tests stay excluded.
Freeze both source-selected models before evaluating either on2b. No target
adaptation or target-selected hyperparameters. Report all arms including failures.

This isolates input scaling within a frozen encoder plus linear readout, not
end-to-end language alignment. Even a positive result does not prove a universal
task recognizer.2b has previous development exposure and is in REVE pretraining;
any downstream result remains exploratory. Emotion unchanged; no task expansion.
