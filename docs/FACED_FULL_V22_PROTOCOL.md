# Expand FACED source subjects v22

After v21's encoder adaptation failed to beat the spectral baseline, increase
source coverage rather than continuing head/loss variations on a16-person
FACED training subset. This is a data-scale experiment, not evidence that more
data will necessarily solve transfer.

Keep the original full123-person split from v14, seed1401:73 train,25 validation,
25 reserved test. Acquire all98 source train/validation subjects only. The
existing16 train+8 validation pilot is a subset with unchanged split membership;
74 additional subjects are required. No subject is chosen based on target
results or substituted because of download/format difficulty. Retain the same
pinned NEMAR revision and license metadata. Expected window count is98*280=27440.

New root: artifacts/faced_full_v22. Reuse existing audited pilot raw/prepared
subject directories through symlinks, only reading completed artifacts; record
pilot audit hashes. Do not alter v14 cohort, metrics or files. New records use
range-checked resumable downloads, full-file SHA256 validation, unchanged trial
labels/onsets and ten3second windows/trial. Unknown references/units and EOG
handling remain documented; acquisition alone does not authorize training.

Four I/O workers. Errors are retained by subject; no blind retries or replacement
cohort. Atomic progress snapshots record actual completion. A complete integrity
audit must verify all98 subjects, split membership, per-window hashes, uniqueness,
finite arrays and class/trial counts before downstream cache extraction.

Downstream comparison must keep validation identical across pilot-only and
expanded training arms. Use the full fixed25-person FACED validation cohort plus
the same SEED validation subjects for both, and retain ordinary spectral and
frozen-encoder baselines. A pilot model selected on8 subjects cannot be directly
compared to a full-data model selected on25 without a matched-selection control.
First compare source validation; freeze all models before any further exploratory
target evaluation. MI coverage remains unchanged. No task expansion yet.

The initial scale controls are four source-only linear spectral runs: pilot/full
FACED training crossed with full/spatially-centered common30-channel fractions.
SEED training remains4050 windows. Both scales use the same8350 validation rows
(SEED1350 + FACED7000); each completion file stores their ordered-row hash and
the controller requires all four to match. Expected training counts are8530
and24490. Source training uses equal dataset/class weight and the unchanged
C grid. No target evaluation is automatically launched after these controls.

## Quality-rule amendment before any v22 model fitting

Subject100 failed the unchanged flat-channel guard. Raw BDF reinspection with
the same filter/window locations identified13 windows with PO3 std<1e-12volts,
including nearly constant values around8e-17. All13 are negative-class windows
in trials4,10,7. SHA256 verified; retrying network transfer cannot repair this.
Evidence: `artifacts/faced_full_v22/sub100_flat_audit.json`.

Opt-in full-cohort policy now quarantines such flat windows with exact trial,
start, label, offending channel and measured std. No replacement window, subject
removal, interpolation or threshold relaxation. Strict mode remains the default
for old workflows. Nonfinite values still fail; every subject must retain all
three classes. Audits require retained+quarantined=280 per subject, unique
candidate locations, original class counts120/40/120 and28trials×10candidates.
Actual model counts may be lower than nominal counts above; dataset/class
sampling weights are recomputed from retained rows. This quality exclusion may
be label-correlated and is disclosed rather than treated as missing at random.

Subject100 now has267 valid windows and13 explicit quarantines; its original
files and cohort membership are unchanged. The current live downloader had
already imported strict code and retains its original failure record. Let it
finish other subjects; only after terminal status, resume the same cohort using
the corrected code to reconcile already repaired records. Do not restart a live
download or erase historical errors. Complete full-cohort integrity validation
is still required before feature extraction and training. Pilot/full controls
must use the same actual validation rows, checked by hash.
