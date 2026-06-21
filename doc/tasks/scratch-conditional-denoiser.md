# Scratch Conditional Denoiser

## Goal

Implement the random-initialized conditional U-Net that predicts diffusion noise from noisy 64x64 RGB tensors, timesteps, and animal/object/pair condition IDs.

## Inputs

- `doc/proposal.md`: Model is a scratch conditional DDPM denoiser with timestep and animal/object conditioning.
- `doc/high-level-design.md`: Scratch Conditional Denoiser owns trainable architecture and must not load pretrained or existing weights.
- `doc/detailed-design.md`: Defines `UNetConfig`, `ConditionalUNet`, forward inputs/outputs, conditioning, CFG null-token support, and no filesystem/network behavior in constructors.
- `doc/test-plan.md`: Requires random initialization, output shape checks, condition/timestep influence, finite gradients, and no checkpoint/pretrained loads.

## Write Scope

May create or edit `scripts/model.py` and `tests/test_model.py`. May add tiny tensor-only fixtures inside tests.

## Read Scope

Inspect `doc/detailed-design.md` Scratch Conditional Denoiser section, `scripts/brainrot_data.py` condition ID contract, and `scripts/diffusion.py` interfaces if they exist.

## Dependencies

Depends on shared condition ID contracts from Data and Condition Registry. Does not depend on diffusion, training, generation, scoring, or packaging.

## Tasks

- [x] Define `UNetConfig` with image channels, base channels, channel multipliers, residual depth, embedding dimension, dropout, label counts, and CFG null-token settings.
- [x] Implement sinusoidal timestep embeddings and learned animal/object/pair embeddings without loading files or external model weights.
- [x] Implement a scratch U-Net forward path that preserves `[batch, 3, 64, 64]` shape and accepts condition ID tensors.
- [x] Validate tensor shapes and label ranges with clear errors.
- [x] Add tests that compare outputs across changed timesteps/conditions and confirm at least one parameter receives finite gradients.
- [x] Add a no-pretrained regression test that constructs the model without touching `.pth`, `.pt`, OpenCLIP, TorchVision pretrained weights, or external checkpoints.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_model`
- [x] `python -m compileall scripts`
- [x] Importing `scripts.model` performs no downloads, filesystem reads, CUDA initialization, training, or generation.

## Done When

- [x] `ConditionalUNet` can be constructed from config and run on a tiny batch.
- [x] Forward output shape exactly matches input image shape.
- [x] Model smoke tests pass from random initialization with finite gradients.
