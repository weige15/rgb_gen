# Output Validator

## Goal

Implement `scripts/validate_outputs.py` as the pass/fail contract checker for generated images before scoring, packaging, or upload.

## Inputs

- `doc/proposal.md`: Output checker must fail unless generated images are exactly 2,000 files, filenames match `dataset/generate.csv`, every file is PNG, and every image is 64x64 RGB.
- `doc/high-level-design.md`: Output Validator owns generated-output contract checks and is a gate before scoring/submission.
- `doc/detailed-design.md`: Defines CLI flags, expected filename set behavior, image inspection algorithm, and actionable error handling.
- `doc/test-plan.md`: Requires pass/fail fixtures for exact count, missing files, extra files, wrong extension, invalid PNG, wrong dimensions, wrong color mode, and filename mismatch.

## Write Scope

May create or edit `scripts/validate_outputs.py`, `tests/test_validate_outputs.py`, and small temporary PNG/invalid-image fixtures under test-controlled temp directories.

## Read Scope

Inspect `dataset/generate.csv`, `scripts/brainrot_data.py` prompt/CSV parsing helpers if present, `doc/detailed-design.md`, and `doc/test-plan.md` validator cases.

## Dependencies

Depends on Data and Condition Registry CSV parsing contracts. Used by Generation Orchestrator, 2,000-Image Scoring Adapter, and Submission Packager.

## Tasks

- [x] Implement CLI parsing for `--generate_csv`, `--image_dir`, `--expected_count`, and `--strict_prompt`.
- [x] Derive expected filename set directly from `generate.csv` and reject bad prompt/category rows when strict mode is enabled.
- [x] Compare actual direct PNG filenames with expected filenames using set equality, not just count.
- [x] Inspect every expected file with PIL and require PNG format, `(64, 64)` size, and RGB mode.
- [x] Report missing files, extra files, invalid images, wrong dimensions, and wrong modes with actionable errors and nonzero exit status.
- [x] Add fixture-based tests for passing outputs and each documented failure case.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_validate_outputs`
- [x] `python -m compileall scripts`
- [x] Validator tests use temporary directories and never inspect or modify final `generated_images/`.

## Done When

- [x] Validator passes a controlled good fixture and fails every controlled bad fixture.
- [x] Final output contract can be checked with `python scripts/validate_outputs.py --generate_csv dataset/generate.csv --image_dir generated_images --expected_count 2000`.
- [x] Tests verify filename set equality and RGB/64x64/PNG inspection.
