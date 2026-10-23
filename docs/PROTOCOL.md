# MI + SEED task-conditioned alignment → JEV

## Current run

One current two-stage approach, restricted to physical GPUs 2 and 3. Entry: `scripts/run_joint_training.py`; configuration: `configs/mi_emotion_train.json`. The controller runs both distributed smoke checks before the production stages and stops on any failure. It never overwrites output directories. Current state is in `artifacts/mi_emotion/pipeline.json`.

## Data

Nine MI sources retain their audited subject-disjoint split: 49,193 train trials and 12,508 validation trials. All exported EEG channels and the full task epoch are retained. The 6,520 2b trials remain outside training and selection. See [MI_DATASETS.md](MI_DATASETS.md).

SEED: 9/3/3 subjects for train/validation/test, corresponding to 405/135/135 movie trials. Training contains 22,734 nonoverlapping four-second windows. SEED-VII: 12/4/4 subjects, 960/320/320 movie trials, 41,844 training windows. Seed 42 fixes subject splits before windowing; all sessions stay with the subject. Validation/test use eight fixed, evenly distributed nonoverlapping windows per parent trial: 1,080 SEED and 2,560 SEED-VII windows each. Thus the fixed validation set has 16,148 signal inputs total. Parent-trial emotion predictions average their eight window probabilities. Do not equate windows with independent trials.

Both emotion datasets use mounted preprocessed 200-Hz signals in microvolts. SEED labels are read from label.mat; SEED-VII labels are read from the provided emotion_label_and_stimuli_order.xlsx. No assumptions based on numeric class ordering in previous scripts. Channel-centering, clip±1000 microvolts, divide by100, then right-edge padding to REVE's 200-sample/180-stride patches. Four seconds is padded from800 to920 samples, yielding62×5=310 tokens. Keep all62 channels with the agreed [SEED coordinate policy](SEED_COORDINATES.md).

Source-data manifests, prepared float32 trial arrays and splits are ignored under `datasets/mi_emotion/`. `complete.json` contains split counts, subject membership and hashes. Preparation validates finite/nonflat signals and no within-dataset subject overlap across splits. Cross-series subject identity and stimulus overlap are not certified; results must not be described as fully independent cross-series generalization.

## Task text and sampling

Catalog: `configs/mi_emotion_tasks.json`. MI has left/right hand; SEED has negative/neutral/positive valence; SEED-VII has happy/neutral/disgust/sad/anger/fear/surprise. Coarse valence and discrete emotion are distinct task-conditioned targets. Input texts contain task and class, not source dataset, subject, session or true-answer hints.

Current contrast training traverses every MI training trial once per shuffled epoch:49,193 unique trials,32 per global batch,1,538steps. The final batch retains9 MI trials without duplication or dropping. Each batch additionally samples SEED16 and SEED-VII16. JEV retains its separate task-balanced sampling until explicitly restarted. Full batches distribute16/8/8 per GPU; the final partial batch distributes the nine MI trials across both ranks. MI sampling is a uniform shuffled permutation of the full manifest, preserving its actual dataset proportions. Emotion sampling remains class-balanced, parent-trial-uniform then window-uniform with replacement. One complete contrast epoch draws49,193 distinct MI trials and24,608 windows from each emotion task. Seeds fix the permutation and emotional samples.

## Stage 1

Original REVE-base with all active encoder parameters trained; unused attention-pooling query frozen. Mean all real channel/time tokens → Linear512→1024 → L2 normalization. Freeze local Qwen3-Embedding-0.6B and embed all12 task+class texts. Symmetric multi-positive InfoNCE uses same(task,class) positives, independently within each task; repeated texts are never negatives. Weighted loss=0.5MI+0.25SEED+0.25SEED-VII. Differentiable global gather then one parameter-gradient average across ranks.

One complete MI epoch (1,538steps), global batch up to64 (last41), microbatch4, AdamW encoder1e-5/head1e-3,80-step linear warmup, temperature0.07, weight decay0.01, gradient clip1. Validate every200steps and at completion. Report each dataset's accuracy and BA, then select by equal macro average of the three task scores (MI score itself averages its nine datasets). Emotion selection uses parent-trial BA. The pretrained head is used only for contrastive classification and discarded before JEV.

## Stage 2

Freeze the stage1 validation-selected REVE. Feed all real encoder tokens through the connector to the original JEV language backbone. Enable and train original JEV LoRA, connector and native A–P decision rows; Qwen base frozen. Mask output slots beyond each sample's2/3/7 valid options. Every sampled occurrence permutes options independently of the label and updates target position. Loss is ordinary option-position cross entropy, with task weights induced by batch quotas. No attribute questions.

1,600steps, global64, training microbatch4 per GPU, validation microbatch8 per GPU with token-length sorting, constant learning rates connector1e-4/LoRA5e-6/head1e-5. Fixed validation every200steps; evaluate original and reversed option ordering on exactly the same signals. Report semantic accuracy and permutation consistency per dataset. Frozen encoder tokens are computed from raw signals and cached in CPU memory; no old frozen-REVE features are reused.

All testing remains untouched during training; this launch does not automatically tune on or repeatedly score 2b/emotion test sets. Current experiment establishes supervised multitask learning and option binding, not unseen-class recognition.

## Version history

Previous MI-only CE/contrast controls were stopped by the user. Their implementation is archived in Git (0f686ad); ignored checkpoints/logs are retained. Superseded training entry points are removed from the working tree. Current measurements and paths: [RESULTS.md](RESULTS.md).

## Node restart and checkpoint cadence

The first JEV run stopped at logged step173 when the node was lost; it had no saved production JEV checkpoint. Preserve that run under `artifacts/mi_emotion/jev`. The replacement run is `artifacts/mi_emotion/jev_restart`, initialized from the completed contrast best encoder and original JEV weights, starting at step1. Training global batch64 and task quotas are unchanged. Save `last.pt` (weights plus optimizer) after step1 and every50steps, before scheduled validation. Full validation remains every200steps; `best.pt` updates after validation improvement. Two-GPU maximum-length checks test training microbatch4 and validation microbatch8 before launch.

Restart command: `laya/.venv-eeg/bin/python scripts/run_joint_training.py --jev-only --jev-output jev_restart --jev-smoke-output jev_batch_check`. This skips the already completed contrast stage and refuses to overwrite outputs.

## Current contrast-only rerun

The user stopped the prior job. Restart contrast from original REVE and a fresh projection, preserving text templates, task loss weights and optimizer settings. Do not automatically start JEV. Command: `laya/.venv-eeg/bin/python scripts/run_joint_training.py --contrast-only --contrast-output contrast_full_mi --contrast-smoke-output contrast_full_mi_smoke`. Output: `artifacts/mi_emotion/contrast_full_mi/`. Coverage audit: `artifacts/mi_emotion/full_mi_coverage.json`. The distributed smoke tests the final41-sample batch.
