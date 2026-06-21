# Extra Data Intake and Provenance

## Goal

Implement the optional extra-data manifest path that defaults to no external data and only admits records with complete provenance, valid labels, and valid images.

## Inputs

- `doc/proposal.md`: Extra data is allowed but should come after a working baseline and must not add reproducibility burden without benefit.
- `doc/high-level-design.md`: Extra Data Intake and Provenance admits only assignment-allowed, reproducible, manually/provenance-labeled samples.
- `doc/detailed-design.md`: Defines minimum manifest columns, fail-closed validation, and `load_extra_records` / `validate_extra_manifest` interfaces.
- `doc/test-plan.md`: Requires default-disabled behavior and rejection of unknown labels, missing provenance, non-reproducible paths, wrong dimensions, pretrained-generated images, and pretrained-generated labels.

## Write Scope

May create or edit extra-data helpers in `scripts/brainrot_data.py` or a small `scripts/extra_data.py` if splitting is simpler, plus `tests/test_extra_data.py` and temporary fixture manifests/images.

## Read Scope

Inspect `scripts/brainrot_data.py`, `doc/detailed-design.md` shared contracts, `doc/test-plan.md` extra-data cases, and any user-provided extra-data manifest.

## Dependencies

Depends on Data and Condition Registry for category mapping, image validation, and `TrainRecord` construction.

## Tasks

- [x] Implement a no-manifest path that returns an empty extra-record list.
- [x] Validate manifest columns: `image_path`, `animal`, `object`, `source`, `retrieval_date`, `license_or_allowance`, `label_author_or_method`, `image_hash`, and `preprocessing_notes`.
- [x] Reject unknown labels, missing provenance fields, unreadable paths, invalid image dimensions/modes, pretrained-model-generated images, and pretrained-model-generated labels.
- [x] Return validated extra records using the same condition and image contracts as repository training data.
- [x] Add `unittest` fixtures for accepted manifests and every documented rejection path.
- [ ] Document in README only after implementation that external extra data is disabled by default.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_extra_data`
- [x] `python -m unittest tests.test_brainrot_data`
- [x] Extra-data tests do not download files and do not require network access.

## Done When

- [x] Training can call the extra-data loader safely when no manifest is supplied.
- [x] Invalid extra-data manifests fail before training starts.
- [x] Tests prove pretrained-generated/labeled extra data is rejected from the generator training path.
