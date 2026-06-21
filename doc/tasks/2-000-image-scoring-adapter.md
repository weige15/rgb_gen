# 2,000-Image Scoring Adapter

## Goal

Implement `scripts/score_2000.py` as a development-only scorer for exactly the generated 2,000-image set, computing FID and CLIP-T without raw validation images and without influencing generation.

## Inputs

- `doc/proposal.md`: Local scorer files are incomplete, official target is 2,000 images, and Codabench remains authoritative.
- `doc/high-level-design.md`: Scoring adapter targets generated 2,000 images, uses reference statistics for FID and prompts for CLIP-T, and keeps evaluator-only pretrained models outside the generator path.
- `doc/detailed-design.md`: Defines CLI flags, score report fields, FID/CLIP-T algorithms, evaluator-only dependency isolation, and failure handling.
- `doc/test-plan.md`: Requires missing-fixture failures, exact 2,000 target checks, no raw validation image requirement for CLIP-T, no unrun verified-score claims, and isolation from training/generation.

## Write Scope

May create or edit `scripts/score_2000.py`, `tests/test_score_2000.py`, and tiny test fixtures under test-controlled temp directories. Must not modify `scoring_program/score.py` or write `scoring_program/scores.json` during tests.

## Read Scope

Inspect `scoring_program/score.py`, `scoring_program/input/ref/test_mu.npy`, `scoring_program/input/ref/test_sigma.npy`, `dataset/generate.csv`, `scripts/validate_outputs.py`, and scorer caveats in `doc/quality-gates.md`.

## Dependencies

Depends on Output Validator and Data and Condition Registry prompt parsing. Uses evaluator-only NumPy, SciPy, PyTorch, TorchVision, OpenCLIP, PIL, and tqdm dependencies when scoring is actually run.

## Tasks

- [x] Implement CLI parsing for `--image_dir`, `--generate_csv`, `--ref_mu`, `--ref_sigma`, `--scores`, `--output`, `--device`, `--batch_size`, `--num_workers`, and `--overwrite`.
- [x] Call or mirror Output Validator so scoring refuses non-2,000 final outputs.
- [x] Implement FID using generated image Inception features plus `test_mu.npy`/`test_sigma.npy`.
- [x] Implement CLIP-T using generated images and prompts from `generate.csv`, without requiring raw validation images or CLIP-I fixtures.
- [x] Write a score JSON report with metric values, command metadata, evaluator model identifiers, and notes that Codabench is authoritative.
- [x] Add tests for missing reference statistics, missing prompts, output-file overwrite protection, exact target count enforcement, and training/generation import isolation.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_score_2000`
- [x] Scoring tests do not download evaluator weights unless explicitly marked as an integration/manual check.
- [x] `scripts.train` and `scripts.generate` do not import `scripts.score_2000`.

## Done When

- [x] The adapter has a clear planned command for scoring generated 2,000-image outputs.
- [x] Missing scorer inputs fail clearly before model loading.
- [x] Tests prove evaluator-only pretrained weights are isolated from the generator path.
