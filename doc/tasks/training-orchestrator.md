# Training Orchestrator

## Goal

Implement `scripts/train.py` as the from-scratch training entry point that validates data/devices, trains the denoiser with diffusion loss, logs progress, and saves `model.pth` without unsafe overwrites.

## Inputs

- `doc/proposal.md`: Training must be reproducible, use random initialization, save `model.pth`, track loss/sample grids, and avoid pretrained/existing model weights.
- `doc/high-level-design.md`: Training Orchestrator owns lifecycle, EMA, progress/logging, single-GPU and `torchrun` multi-GPU support up to 4 RTX 3090 GPUs.
- `doc/detailed-design.md`: Defines planned CLI flags, checkpoint contract, output overwrite behavior, atomic `model.pth` save, DDP handling, and progress log fields.
- `doc/test-plan.md`: Requires one-step training smoke, finite gradients, seed setup, no hardcoded `cuda:0`, max 4-GPU validation, progress logs, and no final-output writes.

## Write Scope

May create or edit `scripts/train.py`, shared small runtime helpers if needed, `tests/test_train.py`, and temporary test output fixtures under test-controlled temp directories. Must not write real `model.pth`, `checkpoints/`, or `runs/` during tests except through temp paths.

## Read Scope

Inspect `scripts/brainrot_data.py`, `scripts/model.py`, `scripts/diffusion.py`, `dataset/train.csv`, `dataset/trainset/`, `doc/detailed-design.md`, and existing tests.

## Dependencies

Depends on Data and Condition Registry, Extra Data Intake and Provenance, Scratch Conditional Denoiser, and Diffusion Process and Sampler.

## Tasks

- [x] Implement CLI parsing for the detailed-design training flags with safe defaults pointing at repository dataset paths.
- [x] Add seed setup, output-path overwrite checks, device validation, max 4-GPU enforcement, and no hardcoded `cuda:0`.
- [x] Build the dataset, optional extra-data records, model, diffusion process, optimizer, and optional EMA state from serializable config.
- [x] Implement single-process training and `torchrun`/DDP initialization paths with progress and persistent log fields.
- [x] Save `model.pth` atomically with model/diffusion config, ordered labels, seed, train step, and `uses_pretrained_generator_weights: False`.
- [x] Add a one-step temp-directory training smoke test with finite loss/gradients and no writes to final generated artifact paths.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_train`
- [x] `python -m unittest tests.test_brainrot_data tests.test_model tests.test_diffusion`
- [x] `python -m compileall scripts`
- [x] Training smoke confirms progress/log fields and rejects existing output paths without `--overwrite`.

## Done When

- [x] `scripts/train.py` can run a tiny one-step smoke training path from random initialization.
- [x] Saved checkpoints follow the `model.pth` checkpoint contract.
- [x] Tests prove training does not load pretrained/existing generator weights or write final generated outputs.
