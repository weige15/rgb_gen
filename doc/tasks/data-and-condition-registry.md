# Data and Condition Registry

## Goal

Implement the shared data layer that parses assignment CSVs, validates labels/prompts/images, and exposes stable condition records for training, generation, validation, and packaging.

## Inputs

- `doc/proposal.md`: Correctness strategy requires CSV parsing, 10x10 category mapping, image existence checks, and 64x64 RGB validation.
- `doc/high-level-design.md`: Data and Condition Registry owns CSV parsing, image loading, animal/object/pair mapping, and failures on malformed data.
- `doc/detailed-design.md`: Defines category order, CSV contracts, `Condition`, `TrainRecord`, `GenerationRequest`, and planned `scripts/brainrot_data.py` interfaces.
- `doc/test-plan.md`: Requires parser tests for schemas, deterministic mappings, duplicate IDs, unknown labels, prompt consistency, missing images, and 64x64 RGB checks.

## Write Scope

May create or edit `scripts/brainrot_data.py`, `tests/test_brainrot_data.py`, and small temporary test fixtures under `tests/fixtures/`. May update `README.md` only to document verified data contracts after implementation.

## Read Scope

Inspect `dataset/train.csv`, `dataset/generate.csv`, representative files in `dataset/trainset/`, `doc/detailed-design.md`, and existing tests if present.

## Dependencies

None. This module is the base dependency for training, generation, validation, extra-data intake, scoring, and packaging.

## Tasks

- [x] Define assignment-order animal/object vocabularies and stable `animal_id`, `object_id`, and `pair_id` mapping helpers.
- [x] Add frozen dataclasses for `Condition`, `TrainRecord`, and `GenerationRequest`.
- [x] Implement strict `train.csv` and `generate.csv` parsing with required-column checks, duplicate-ID rejection, unknown-label rejection, and prompt validation.
- [x] Implement PIL-based image inspection/loading that verifies PNG, RGB, and 64x64 before returning normalized tensors for training.
- [x] Implement `BrainrotDataset` with optional class-preserving augmentation that never changes image size or labels.
- [x] Add fixture-based `unittest` coverage for valid rows, invalid rows, duplicate IDs, missing images, wrong dimensions, `coffee cup` prompts, and mapping stability.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_brainrot_data`
- [x] `python -m compileall scripts`
- [x] Importing `scripts.brainrot_data` performs no file writes, downloads, training, generation, or CUDA initialization.

## Done When

- [x] Training and generation CSVs can be parsed into typed records with deterministic condition IDs.
- [x] Bad CSV rows and bad images fail with row/path-specific errors.
- [x] Data registry tests pass without touching `generated_images/`, `checkpoints/`, `runs/`, `model.pth`, or score files.
