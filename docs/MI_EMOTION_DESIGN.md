# MI + SEED joint training design

Status: design only, not yet implemented or running. On 2026-10-22 the two MI JEV arms were stopped at logged step 1200 at the user's request. Saved checkpoints/logs remain intact; validation files currently end at step 1000. Previous runnable implementation is preserved in Git at 0f686ad. Task definitions: [mi_emotion_tasks.json](../configs/mi_emotion_tasks.json).

## Data and task boundaries

Retain the nine MI sources and their existing subject-disjoint training/validation split (49,193/12,508 trials). Keep 2b out of training. Add mounted SEED (negative/neutral/positive) and SEED-VII (happy/neutral/disgust/sad/anger/fear/surprise). SEED-IV/V have not been found in the inspected locations. SEED-DV is not assumed to be an emotion dataset.

SEED and SEED-VII are distinct tasks: coarse negative valence is not interchangeable with sadness, fear, anger or disgust. Text contains task and class, never the dataset name or subject/session ID. SEED-VII's supplied workbook has four sessions with only five of the seven classes per session; always offer all seven options, not a session-dependent subset.

Before preparing caches, verify raw label mapping, sampling rate, physical units, all 62 electrode names and coordinates against supplied metadata. Do not silently drop unmapped electrodes. Existing legacy emotion caches are not accepted without re-audit. Split subjects before windowing, keep all sessions/windows of each subject together, and check cross-series subject/stimulus overlap before making independent-test claims. Freeze split manifests before any training. Use fixed nonoverlapping 4-second emotion windows; weight/sample parent trials uniformly so longer videos do not dominate. Exact window counts and subject IDs must be published after preparation, not inferred from file sizes.

## Stage 1: EEG/text contrastive pretraining

Start from original REVE, train its encoder and a semantic projection, and freeze Qwen3-Embedding-0.6B. Do not train only against numeric class IDs. IDs merely index actual task+class text embeddings. Examples:

- Task: motor imagery recognition. Class: Left hand motor imagery.
- Task: emotional valence recognition. Class: Negative emotion.
- Task: discrete emotion recognition. Class: Sadness.

Use symmetric multi-positive InfoNCE (temperature 0.07), with positives defined by the same (task, class) pair. Repeated class descriptions must not become negatives. Compute loss separately within each task and combine 0.5 MI + 0.25 SEED + 0.25 SEED-VII: this prevents easy cross-task separation from substituting for within-task class discrimination. The two neutral descriptions remain separate task-conditioned targets; they are not forced apart by cross-task negatives.

For global batch 64, sample 32 MI, 16 SEED, 16 SEED-VII. Within each task balance semantic classes over time; within MI preserve diversity across source datasets and subjects. Record actual sample draws separately from unique trials/windows. Train for a short bounded schedule first; choose duration from the audited sample inventory and measure throughput before quoting an ETA. Select using task-macro validation balanced accuracy and also report every task independently.

## Stage 2: shuffled-option JEV

Freeze the selected REVE, use all its valid tokens through the trainable connector, and train the enabled JEV LoRA and decision head with option-position cross entropy. Qwen base stays frozen. Initialize from original JEV weights rather than a head permanently associated with left/right hand. Support seven output positions A–G, initialized from the matching original JEV choice rows. The existing implementation is only binary and must be generalized before running this plan.

Use a shared seven-slot head, masking unused slots before softmax: MI activates two slots, SEED three, SEED-VII seven. Randomly permute all options for every sampled example and regenerate the target index. Permutations must be independent of class, subject, dataset and signal, reproducible from recorded seeds. Text tells the model the task and lists candidates, but never supplies the correct class. Classification targets are stimulus/task annotations, not claims of directly observed subjective emotion.

Example for a negative SEED window:

```
[kind] choice
[state] EEG recording begins:
<EEG embeddings>
EEG recording ends.
[question] What is the emotional valence associated with this EEG recording?
[options]
A) Positive emotion
B) Negative emotion
C) Neutral emotion
[decision]:
```

The training target is B. Another presentation of the same signal may place Negative emotion at A or C and must change the target accordingly. Use the same task sampling weights as stage 1. No auxiliary spectral/attribute questions in this experiment.

## Validation and claims

Report MI per dataset, SEED and SEED-VII separately: accuracy, balanced accuracy, confusion matrices, class recall and sample counts. For emotion report window-level and parent-trial-level results separately, with trial probabilities averaged over fixed windows. Never mix window and trial denominators. For every task evaluate fixed permutations on the identical held-out signals, convert answer positions back to semantic classes, and report accuracy plus semantic consistency. Consistency alone is insufficient because a constant-class predictor is also consistent.

Report a task-macro selection score separately from per-task results. Use within-task full candidate sets as primary evaluation. Mixed MI/emotion distractors are an additional diagnostic with the task explicitly specified; task-name recognition alone must not count as successful signal classification. This run tests supervised multitask learning and option binding, not unseen-emotion zero-shot generalization. Leave-class-out evaluation requires a separate predeclared split.

## Implementation gates

1. Audit SEED metadata, coordinates and raw signals; create reproducible subject/trial/window manifests in ignored datasets storage.
2. Generalize text prototypes, task-specific loss, task-balanced sampler, variable-choice prompts and head masking. Add tests for permutation/target consistency and masked-slot exclusion.
3. Run small forward/backward and per-task overfit checks, then the joint contrast stage followed by JEV. Keep checkpoints, metrics and dataset manifests outside tracked code.
