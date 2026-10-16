# Repository maintenance

- Keep one current experimental approach in the working tree. Necessary matched controls and tests belong to that approach.
- When replacing an approach, commit the prior state, remove superseded implementation/configuration/docs, and commit the replacement. Do not accumulate vN copies of independent experiment code.
- Git history is the archive. Use isolated worktrees to inspect or rerun historical versions.
- Do not delete ignored datasets, checkpoints, logs, caches, environments, or third-party installations as part of code cleanup: they are not recoverable from Git.
- Before changing code used by a live job, check its imports and hash guards. Preserve active training semantics and required files until the job finishes; minimize compatibility modules to the current helpers.
- Keep README and current protocol/results consistent with the active implementation. Do not retain dangling links to removed experiment documents.
