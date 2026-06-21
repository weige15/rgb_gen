# Generation Orchestrator

## Goal

Implement `scripts/generate.py` to load `model.pth`, read `dataset/generate.csv`, sample deterministic request-aligned images, and write RGB PNG outputs without unsafe overwrites.

## Inputs

- `doc/proposal.md`: Generation must write exactly 2,000 64x64 PNG files matching `dataset/generate.csv` IDs.
- `doc/high-level-design.md`: Generation Orchestrator owns request ordering, progress/logging, single-GPU and explicit multi-GPU generation up to 4 GPUs, and no pretrained/existing model use.
- `doc/detailed-design.md`: Defines CLI flags, checkpoint compatibility checks, per-request seed behavior, request partitioning, RGB PNG saves, and output collision handling.
- `doc/test-plan.md`: Requires temporary generation smoke, deterministic filenames, same-device seed repeatability, multi-GPU partition coverage, overwrite protection, and no-pretrained checks.

## Write Scope

May create or edit `scripts/generate.py`, shared runtime helpers if needed, `tests/test_generate.py`, and temporary generated-image fixtures in test temp directories. Must not write final `generated_images/` during tests.

## Read Scope

Inspect `scripts/brainrot_data.py`, `scripts/model.py`, `scripts/diffusion.py`, `scripts/validate_outputs.py` if present, checkpoint contract in `doc/detailed-design.md`, and `dataset/generate.csv`.

## Dependencies

Depends on Data and Condition Registry, Scratch Conditional Denoiser, Diffusion Process and Sampler, and the checkpoint contract produced by Training Orchestrator. Integrates with Output Validator after outputs exist.

## Tasks

- [x] Implement CLI parsing for `--model`, `--generate_csv`, `--output_dir`, seed, sampler, steps, guidance scale, batch size, EMA, devices, `--limit`, and `--overwrite`.
- [x] Load `model.pth`, rebuild model/diffusion from checkpoint config, and reject incompatible or missing category mappings.
- [x] Validate output directory collisions and require temp output or `--limit` for smoke runs.
- [x] Partition requests by rank for `torchrun` while preserving per-request seed `base_seed + row_index` and unique filenames.
- [x] Convert sampled tensors to 64x64 RGB PNG files named exactly from generation requests.
- [x] Add tests for temporary generation, filename preservation, overwrite rejection, same-device seed repeatability, partition no-duplicate/no-missing behavior, and no pretrained/existing model usage.

## Tests and Quality Gates

- [x] `python -m unittest tests.test_generate`
- [x] `python -m unittest tests.test_brainrot_data tests.test_model tests.test_diffusion`
- [x] Generation smoke writes only to a temporary output directory and can be validated by Output Validator when implemented.

## Done When

- [x] A small temporary generation run produces valid PNGs with requested filenames.
- [x] Multi-rank partition logic covers every request exactly once.
- [x] Tests prove generation does not load pretrained models, filter/rerank images, or write final outputs without explicit user action.
