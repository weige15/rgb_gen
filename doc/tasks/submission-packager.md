# Submission Packager

## Goal

Implement `scripts/package_submission.py` to assemble the E3 archive only after required artifacts exist and generated images pass validation.

## Inputs

- `doc/proposal.md`: Final deliverables include `generated_images/`, `scripts/`, `model.pth`, `README.md`, and `requirements.txt`.
- `doc/high-level-design.md`: Submission Packager is optional, consumes final artifacts after validation, and names the archive `HW6_314511048.zip`.
- `doc/detailed-design.md`: Defines CLI flags, required archive entries, standard-library `zipfile` design, exclusion of development-only directories, and failure handling.
- `doc/test-plan.md`: Requires archive naming, required top-level entries, missing-file failures, and no accidental inclusion of checkpoints/runs unless intended.

## Write Scope

May create or edit `scripts/package_submission.py`, `tests/test_package_submission.py`, and temporary archive fixtures under test-controlled temp directories.

## Read Scope

Inspect `scripts/validate_outputs.py`, `generated_images/` only when explicitly packaging final outputs, `README.md`, `requirements.txt`, `scripts/`, `model.pth`, and `doc/detailed-design.md` packager contract.

## Dependencies

Depends on Output Validator and final artifacts from Training Orchestrator, Generation Orchestrator, Documentation and Dependency Manifest, and implemented scripts.

## Tasks

- [x] Implement CLI parsing for `--student_id`, `--generated_dir`, `--scripts_dir`, `--model`, `--readme`, `--requirements`, `--output`, and `--overwrite`.
- [x] Validate required paths exist and reject missing `model.pth`, `README.md`, `requirements.txt`, `scripts/`, or generated images.
- [x] Invoke or reuse Output Validator before creating the archive.
- [x] Create `HW6_314511048.zip` with required top-level entries using standard library `zipfile`.
- [x] Exclude `checkpoints/`, `runs/`, score reports, and other development-only artifacts unless a later task explicitly changes scope.
- [x] Add tests for correct archive name, missing-file failures, expected top-level contents, validation failure propagation, and overwrite protection.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_package_submission`
- [x] `python -m unittest tests.test_validate_outputs`
- [x] Packager tests use temp artifacts and never package real final outputs unless explicitly run by the user.

## Done When

- [x] The packager creates `HW6_314511048.zip` from validated temp fixtures.
- [x] Missing required artifacts fail before archive creation.
- [x] Tests verify required contents and exclusion of development-only directories.
