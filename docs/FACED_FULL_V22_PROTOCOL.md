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
