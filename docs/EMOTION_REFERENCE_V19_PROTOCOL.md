# Amplitude-preserving reference control v19

Local provenance audit: all24 FACED EEG sidecars report EEGReference="n/a"
(15 at250Hz, 9 at1000Hz); SEED Preprocessed_EEG/readme.txt supplies trial labels
but no reference/units specification. We cannot attribute poor transfer to a
known reference mismatch. Metadata Manufacturer values alone are not verified
acquisition history.

Test a controlled alternative: same ordered30 scalp channels, retain their
original amplitude relationships, then compare as_recorded against common
average reference over those30 channels. Exclude mastoid/auxiliary electrodes.
Read SEED MAT amplitude data and FACED BDF via MNE volts, resample200Hz and
bandpass1–30Hz exactly as before. Apply reference before any per-channel
normalization; no zscore or clipping is used in either arm. Extract the same
five log band fractions from identical600-sample windows. Common unit scaling
cancels in fractions. This cannot undo unknown upstream channel-specific
processing in distributed files. An average over30 electrodes is not claimed
to recreate either original acquisition reference.

Four CPU workers extract disjoint recording shards. Preserve exact v16
subjects/windows/labels. Fit full_linear and spatial_linear for each reference
mode, each in both source directions:8 source-selected models. Same C grid,
class balancing, source-only StandardScaler and subject-balanced validation as
v16. Freeze all8 model hashes before held-dataset evaluation. Report both
reference modes, both directions, fixed shuffled predictions and subject CIs.
No target choice of reference mode or classifier, no 2b/SEED-VII evaluation.

The as_recorded arm is newly extracted without clipping in both datasets; its
comparison to v16 is a preprocessing reproducibility check. CAR's causal
contrast is with this new matched as_recorded arm, not a differently processed
old cache. Two tests verify cancellation of a shared reference waveform and
unit invariance of spectral fractions. These are exploratory known-task
emotion diagnostics; the universal zero-shot goal remains unmet.
