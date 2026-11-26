# MedJev source provenance

Source: https://github.com/zfkarl/MedJev

Pinned revision: `386c19c963f2a1a32ea7210e8b30c55ca3c5303b` (user's local clone, 2026-11-26).

The Python/shell source, requirements, README, and MIT license are vendored unchanged. Figures, Git internals, environment files, datasets, and model weights are not copied. README figure links refer to omitted upstream artwork; use the original repository to view them.

EEG-specific integration is in `jev_train/`; this directory remains an upstream reference. Only reviewed model/readout definitions are imported. Upstream medical data loaders and training launchers are not used for EEG. Our batched readout is checked against the upstream single-request readout.

Base weights are supplied separately from the user's `../MedJev/Qwen3.5-4B` download. No medical fine-tuning checkpoint is assumed or silently downloaded.
