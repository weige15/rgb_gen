# Diffusion Process and Sampler

## Goal

Implement the diffusion schedule, forward noising, MSE training loss, CFG mixing, and reverse sampling contract shared by training and generation.

## Inputs

- `doc/proposal.md`: DDPM training uses MSE noise-prediction loss, EMA sampling, classifier-free guidance, cosine schedule, and DDIM-style sampling as possible improvements.
- `doc/high-level-design.md`: Diffusion Process and Sampler owns schedule, loss, and sampling while the denoiser owns model weights.
- `doc/detailed-design.md`: Defines `DiffusionConfig`, `GaussianDiffusion`, `training_loss`, `sample`, schedule validation, DDPM-first implementation, and formula-level behavior.
- `doc/test-plan.md`: Requires schedule properties, direct `q_sample` formula checks, scalar finite loss, seeded sampling determinism, and pixel-range conversion checks.

## Write Scope

May create or edit `scripts/diffusion.py` and `tests/test_diffusion.py`. May add tiny tensor-only fixtures inside tests.

## Read Scope

Inspect `scripts/model.py`, `doc/detailed-design.md` Diffusion section, and test-plan diffusion oracle requirements.

## Dependencies

Depends on Scratch Conditional Denoiser interface for model calls and condition input shape. Does not depend on training or generation orchestrators.

## Tasks

- [x] Define `DiffusionConfig` and validated schedule construction for linear and cosine schedules.
- [x] Precompute finite schedule tensors with monotonic cumulative alphas and device-safe tensor access.
- [x] Implement `q_sample` and `training_loss` against an epsilon-prediction target.
- [x] Implement DDPM sampling first, with deterministic generator support and CFG conditional/unconditional mixing.
- [x] Add pixel conversion/clamping helper for generation-time image saves.
- [x] Add formula-oracle `unittest` coverage for `q_sample`, schedule validity, scalar finite loss, seeded same-device repeatability, and invalid sampler/schedule failures.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_diffusion`
- [x] `python -m unittest tests.test_model`
- [x] DDPM sampling smoke uses tiny tensors and does not write final generated outputs.

## Done When

- [x] Training loss returns a finite scalar on a tiny batch.
- [x] Sampling returns finite tensors with expected shape and clamped image conversion.
- [x] Diffusion tests pass without using pretrained models or generated artifact directories.
