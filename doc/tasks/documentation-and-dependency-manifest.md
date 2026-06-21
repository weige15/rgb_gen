# Documentation and Dependency Manifest

## Goal

Create the reproducibility documentation and pip dependency manifest needed for teaching assistants to install, train, generate, validate, score, and package the project.

## Inputs

- `doc/proposal.md`: Final reproducibility requires commands, seeds, `model.pth`, `README.md`, and `requirements.txt`.
- `doc/high-level-design.md`: Documentation and Dependency Manifest records setup, training, generation, validation, scoring, seeds, hardware, and dependencies.
- `doc/detailed-design.md`: Defines README/requirements responsibilities, planned commands, command-labeling rules, overwrite caveats, and no-pretrained documentation requirements.
- `doc/test-plan.md`: Requires README review, pip compatibility, absence of prohibited high-level generation pipeline dependencies, and accurate labeling of run/planned/blocked commands.

## Write Scope

May create or edit `README.md`, `requirements.txt`, and documentation-only notes needed to keep commands reproducible. May update `AGENTS.md` only if build/test/lint/format/dependency workflows are newly verified.

## Read Scope

Inspect all implemented `scripts/` modules, final test commands, generated logs/configs, `doc/detailed-design.md`, `doc/test-plan.md`, `AGENTS.md`, and scorer caveats in `scoring_manual.txt` / `scoring_program/score.py`.

## Dependencies

Depends on the final implemented CLI flags and dependency choices from all implementation modules. Should be updated after each workflow becomes runnable.

## Tasks

- [x] Add `requirements.txt` with direct pip dependencies needed by implemented training, generation, validation, scoring, and packaging code.
- [x] Update `README.md` with setup, training, generation, validation, optional scoring, packaging, seed, hardware, no-pretrained, and extra-data notes.
- [x] Document commands only after the corresponding scripts exist, and label unrun or blocked commands accurately.
- [x] Document overwrite behavior for `model.pth`, `generated_images/`, `checkpoints/`, `runs/`, score files, and archives.
- [x] Document local scoring caveats: 2,000-image adapter, Codabench authoritative metrics, evaluator-only pretrained weights, and no generator use of scoring models.
- [x] Add documentation review checks for prohibited dependencies, `--score` vs `--scores` confusion, missing seeds, and unsupported metric claims.

## Tests and Quality Gates

- [ ] `python -m pip install -r requirements.txt` after `requirements.txt` exists. Not run: skipped to avoid mutating the active Python environment during this implementation pass.
- [x] `python -m unittest discover` after tests exist.
- [x] `python -m compileall scoring_program scripts` after scripts exist.
- [x] Manual README review confirms no unrun metric claims and no prohibited generator dependencies.

## Done When

- [x] `README.md` explains how to reproduce training and generation from `model.pth`.
- [x] `requirements.txt` is pip-compatible and excludes prohibited high-level generation pipeline dependencies.
- [x] Documentation clearly separates training/generation from evaluator-only scoring.
