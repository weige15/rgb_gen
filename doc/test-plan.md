# Test Plan

## Purpose

This plan defines how to verify the scratch conditional Brainrot image generator before implementation is treated as complete. "Done" means the code can parse the assignment data, train and generate from scratch, validate the 2,000-image output contract, avoid prohibited pretrained/existing models in the generator path, provide progress and logs, support the planned GPU modes, and produce submission artifacts with observable pass/fail checks.

This is a planning artifact. No tests, production code, training runs, generation runs, or evaluator commands were implemented or run while creating this plan.

## Source Requirements

Sources read:

- `doc/proposal.md`: scratch conditional DDPM approach, modules, validation plan, output contract, scoring goals, and assignment constraints.
- `doc/high-level-design.md`: HLD modules, data flow, multi-GPU requirement, progress/logging requirement, 2,000-image scoring adapter, student ID `314511048`, extra-data path, and no-pretrained/no-existing-model policy.
- `doc/problem-brief.md`: assignment objective, dataset schema, allowed/disallowed model usage, deliverables, grading metrics, and threshold scores.
- `doc/repo-map.md`: current repository state, missing scripts/tests/dependency manifest, dataset facts, and scorer fixture gaps.
- `doc/quality-gates.md`: discovered commands, missing quality gates, scorer caveats, and minimum done criteria.
- `AGENTS.md`: local command rules, pip package manager expectation, recommended commands after implementation, and generated-artifact protection rules.
- `scoring_manual.txt`, `scoring_program/metadata`, `scoring_program/score.py`, and `scoring_program/input/ref/config.json`: local scorer behavior and command inconsistencies.

Extracted requirements:

- Generate exactly 2,000 64x64 RGB PNG images named from `dataset/generate.csv`.
- Train the main generator from scratch with no pretrained or existing models in training, generation, conditioning, filtering, or reranking.
- Use PyTorch to implement a scratch conditional DDPM with custom data loading, denoiser, diffusion process, training loop, and sampler.
- Support console progress and persistent logs for training and generation.
- Support single-GPU and explicit multi-GPU execution using at most 4 RTX 3090 GPUs.
- Use the repository `dataset/` as the default training source, with class-preserving augmentation. External extra data is optional and must be assignment-allowed, reproducible, manually/provenance-labeled, and shown to help.
- Score generated 2,000-image sets for FID and CLIP-T during development with a separate scoring wrapper: FID from generated images plus reference statistics, CLIP-T from generated images plus prompts.
- Package final E3 archive as `HW6_314511048.zip`.

Missing or ambiguous source material:

- Exact implementation CLI flags are not defined yet, except that dataset defaults should point at the repository `dataset/` files.
- Exact external extra-data source is not defined; default is no external extra data for the first competitive run.
- Target wall-clock training budget is under 8 hours with 4 RTX 3090 GPUs.
- Local scoring adapter approach is a separate FID/CLIP-T wrapper; detailed design still needs exact command names.

## Test Scope

Covered by this plan:

- CSV parsing for `dataset/train.csv` and `dataset/generate.csv`.
- Animal, object, and pair condition mapping for 10 animals, 10 objects, and 100 pairs.
- Training image loading and 64x64 RGB validation.
- Extra-data intake, provenance, and condition validation, with external extra data disabled by default unless curated.
- Scratch denoiser shape behavior, condition use, and no-pretrained initialization checks.
- Diffusion schedule, noise addition, training loss, sampling shape/range, EMA/CFG/DDIM options when implemented.
- Training orchestration, seed handling, checkpoint/model persistence, progress bars, logs, and multi-GPU launch behavior.
- Generation orchestration, deterministic request ordering, filename preservation, image writing, progress/logging, and multi-GPU partitioning.
- Output validation for count, filenames, PNG format, dimensions, RGB mode, and missing/extra files.
- 2,000-image scoring adapter behavior and local scorer fixture checks.
- Submission packaging for required files and `HW6_314511048.zip`.
- README and dependency manifest verification.

## Non-Tested Scope

Out of scope for this phase:

- Running full training to convergence.
- Running Codabench uploads or using Codabench as part of automated local tests.
- Claiming FID or CLIP-T thresholds are met before generated outputs and scorer inputs exist.
- Verifying hidden test-set performance beyond official Codabench results.
- Modifying `scoring_program/score.py` unless a later scoring task explicitly asks for it.
- Validating extra-data licenses beyond recording provenance and assignment allowance evidence.
- Testing CUDA driver, hardware health, or cluster scheduling outside the project scripts.

## Smoke Tests

| Status | Check | Pass Criteria | Failure Examples |
| --- | --- | --- | --- |
| Planned | Import smoke for future `scripts/` modules | `scripts.brainrot_data`, `scripts.model`, `scripts.diffusion`, training, generation, validation, scoring adapter, and packager modules import without side effects. | Import starts training, creates files, downloads weights, or requires CUDA at import time. |
| Planned | Dataset smoke | Loading a tiny subset from `dataset/train.csv` returns image tensor shape `3x64x64` and condition metadata. | Missing image, wrong image mode, unknown category, malformed row. |
| Planned | Diffusion/model smoke | A tiny batch runs one forward noise-prediction loss and backward pass from random initialization. At least one trainable parameter receives finite gradients. | Shape mismatch, NaN/Inf loss, no gradients, pretrained weight load. |
| Planned | Generation smoke | A temporary generation run for a few requests writes only to a temporary output directory and produces valid 64x64 RGB PNGs with requested filenames. | Writes to final `generated_images/`, filename mismatch, non-PNG output, nondeterministic ordering. |
| Planned | Output validator smoke | A controlled temporary fixture passes when count, filenames, format, and dimensions match; fails when one condition is violated. | Validator only warns, ignores extra files, or accepts wrong dimensions. |
| Missing | Actual smoke command | No smoke command exists yet. | `doc/quality-gates.md` reports smoke tests as missing. |

## Unit Tests by Module

| HLD Module | Status | Planned Verification |
| --- | --- | --- |
| Data and Condition Registry | Planned | Parse schemas exactly: `train.csv` columns `id,animal,object`; `generate.csv` columns `id,animal,object,prompt`. Verify 10 animal categories, 10 object categories, 100 pairs, deterministic mapping, duplicate-ID rejection, missing-image rejection, and 64x64 RGB checks. |
| Extra Data Intake and Provenance | Planned | Default to no external extra data. If enabled, accept only manifest rows with source, reproduction note, image path, animal, object, and explicit allowance/provenance fields. Reject unknown labels, missing provenance, non-reproducible paths, wrong dimensions, pretrained-model-generated images, and pretrained-model-generated labels. |
| Scratch Conditional Denoiser | Planned | Verify random initialization, output tensor shape equals input image shape, condition/timestep inputs affect output, train/eval modes behave consistently, and no checkpoint/pretrained loads occur in constructor. |
| Diffusion Process and Sampler | Planned | Verify schedule ranges, monotonic cumulative alpha behavior, `q_sample` shape preservation, loss scalar behavior, finite outputs, seeded sampling determinism, and pixel range conversion before image save. |
| Training Orchestrator | Planned | Verify seed setup, single-GPU fallback, `torchrun` multi-GPU launch, explicit device list validation with max 4 GPUs, no hardcoded `cuda:0`, progress/log fields, checkpoint/model save policy, and no overwrite without explicit opt-in. |
| Generation Orchestrator | Planned | Verify request ordering, device partitioning, deterministic filenames, no final-output overwrite without opt-in, progress/log fields, and generated count equals request count. |
| Output Validator | Planned | Verify pass/fail cases for exact count, missing files, extra files, wrong extension, invalid PNG, wrong dimensions, wrong color mode, and filename set mismatch. |
| 2,000-Image Scoring Adapter | Planned | Verify it targets exactly the 2,000 submission files, computes FID from generated images plus reference statistics, computes CLIP-T from generated images plus prompts, does not require raw validation images, does not present unrun scores as verified, and separates evaluator-only CLIP/Inception usage from generator training/generation. |
| Submission Packager | Planned | Verify archive name `HW6_314511048.zip`, required top-level entries, no missing `model.pth`, no omitted `requirements.txt`, and no accidental inclusion of checkpoints/runs unless intended. |
| Documentation and Dependency Manifest | Planned | Verify `README.md` has setup, train, generate, validate, scoring, seed, hardware, and reproduction notes. Verify `requirements.txt` exists before install commands are treated as runnable. |
| Existing tests | Missing | No runnable test suite or test directory exists in the current repository. |

## Integration Tests

| Status | Integration Path | Pass Criteria |
| --- | --- | --- |
| Planned | Dataset to training batch | Data registry returns batches consumable by model and diffusion loss without category or shape mismatch. |
| Planned | One-step training integration | Training orchestrator can run one or two optimization steps on a tiny subset, write a temporary checkpoint/model artifact, and log progress without touching final outputs. |
| Planned | Model save/load to generation | A model saved by training smoke can be loaded by generation smoke with compatible architecture and condition mapping. |
| Planned | Generation to validator | Temporary generated files pass the validator when expected count is set for the temporary fixture, and fail when filenames or dimensions are wrong. |
| Planned | Multi-GPU request partitioning | With mocked or available devices, requests are partitioned without duplicate or missing filenames; final output order is independent of worker completion order. Training integration uses `torchrun` semantics. |
| Planned | Scoring adapter fixture check | Adapter refuses to score FID when `test_mu.npy`/`test_sigma.npy` are missing, and refuses to score CLIP-T when prompts are missing from `generate.csv` or a prompt fixture. It does not require raw validation images. |
| Planned | Submission packaging | Packager consumes validated outputs and required files, then produces `HW6_314511048.zip` with expected top-level contents. |
| Missing | End-to-end training and generation command | No training or generation scripts exist yet. Exact commands are Unknown until detailed design. |

## Golden Test Cases

| Status | Case | Expected Behavior |
| --- | --- | --- |
| Planned | Training CSV row `000001.png,fish,chair` | Parser returns image ID `000001.png`, animal `fish`, object `chair`, and a deterministic condition mapping. |
| Planned | Generation CSV row `000001.png,shark,sneaker,a shark and a sneaker` | Parser returns filename `000001.png`, animal `shark`, object `sneaker`, prompt exactly `a shark and a sneaker`, and a condition matching `shark/sneaker`. |
| Planned | Prompt consistency | For every generation row, prompt equals `a {animal} and a {object}` exactly, including multi-word objects like `coffee cup`. |
| Planned | Duplicate generated filename fixture | Validator fails with a duplicate or missing requested filename, even if total PNG count appears correct. |
| Planned | Wrong image dimensions fixture | Validator fails with an actionable error when a PNG is `63x64`, `64x63`, or any size other than `64x64`. |
| Planned | Wrong color mode fixture | Validator fails for grayscale, palette, or RGBA images unless implementation explicitly converts final outputs to RGB before validation. |
| Planned | No-pretrained constructor case | Model construction succeeds from random initialization and does not read `.pth`, `.pt`, OpenCLIP, TorchVision pretrained weights, or external checkpoints. |
| Planned | Diffusion simple tensor case | With fixed tensors, timestep, and supplied noise, `q_sample` returns the formula-defined tensor with exact shape and finite values. |
| Planned | Archive naming case | Submission packager produces `HW6_314511048.zip`, not `HW6_{student_id}.zip` or another placeholder. |

## Oracle or Reference Implementation Strategy

- Data parsing oracle: use Python standard library `csv` and a small hand-written expected list/dict for tiny CSV fixtures.
- Filename oracle: derive expected generated filename set directly from `dataset/generate.csv`; validator output must match set equality, not just count.
- Prompt oracle: compute `f"a {animal} and a {object}"` from each `generate.csv` row and compare to the provided prompt.
- Image oracle: use PIL metadata inspection for dimensions, mode, and PNG format.
- Diffusion oracle: for small tensors, compare `q_sample`, posterior/sampling step helpers, and loss inputs against direct mathematical formulas implemented inside the test, independent of production helpers.
- Randomness oracle: for fixed seeds and fixed runtime settings, compare generated tensors or image bytes for smoke-sized deterministic runs on the same device mode.
- Scoring oracle: Codabench remains authoritative for final FID and CLIP-T. Local FID uses generated image features compared with `test_mu.npy`/`test_sigma.npy`; local CLIP-T uses generated images and text prompts. Local reports are development signals only and must record command, fixture, evaluator weights, model weights, and score file path.
- Model quality oracle: no exact pixel oracle exists for full image quality. Use pass/fail contract tests plus benchmark-only FID/CLIP-T thresholds.

## Randomized or Property Tests

| Status | Property | Bounds and Seed Strategy |
| --- | --- | --- |
| Planned | CSV row order independence for mappings | Shuffle small valid train/generate fixtures with deterministic seeds; mapping and validation results remain stable. |
| Planned | Category validation | Randomly generate valid and invalid animal/object labels; valid labels pass, invalid labels fail with row-specific errors. |
| Planned | Output validator set properties | Randomly remove, duplicate, or add filenames in temporary fixtures; validator fails unless the set exactly matches expected names. |
| Planned | Diffusion schedule properties | Randomly sample timesteps; betas remain in valid range, cumulative alphas are finite and monotonic, and tensor shapes are preserved. |
| Planned | Seed determinism | Repeated smoke generation with the same seed and same device mode produces identical outputs; different seeds produce at least one changed output. |
| Planned | Multi-GPU partition properties | Randomly partition request indices across simulated workers; merged outputs contain every requested filename exactly once. |
| Unknown | Cross-device bitwise determinism | Exact equality across CPU, single GPU, and multi-GPU may not be guaranteed with mixed precision; detailed design must choose acceptable reproducibility criteria. |

## Edge Cases

- Empty `train.csv` or `generate.csv`.
- Missing required CSV columns.
- Extra CSV columns.
- Duplicate training IDs or generation IDs.
- Image ID in CSV missing from `dataset/trainset/`.
- Image file exists but is unreadable or not PNG.
- Training image has wrong dimensions or non-RGB mode.
- Unknown animal or object label.
- Prompt inconsistent with `animal` and `object`.
- Multi-word object labels such as `coffee cup`.
- Less than or more than 20 generation rows for an animal-object pair.
- Missing one of the 100 animal-object pairs.
- Extra-data row with missing provenance, unknown source, invalid condition, pretrained-model-generated image, or pretrained-model-generated label.
- Attempt to load pretrained/existing model weights in training/generation code.
- `generated_images/` already exists and contains files.
- `checkpoints/`, `runs/`, `scores.json`, or `scoring_program/scores.json` already exists.
- Device list includes more than 4 GPUs.
- Device list includes unavailable GPUs.
- Device 0 is busy or unavailable; scripts must not assume `cuda:0`.
- CUDA is unavailable; scripts should either fail clearly or use supported CPU smoke mode where appropriate.
- Scorer reference files missing: `test.json`, `test/`, `test_mu.npy`, or `test_sigma.npy`.
- Local scorer command uses documented `--score` instead of implemented `--scores`.
- Scoring config expects 3,000 images while generated submission set has 2,000 images.
- Packaging before `model.pth`, `requirements.txt`, `README.md`, `scripts/`, or `generated_images/` exists.

## Performance Benchmarks

| Status | Benchmark | Metrics and Thresholds |
| --- | --- | --- |
| Planned | Training throughput smoke | Report images/sec, step time, GPU count, memory usage, loss, elapsed time, and ETA for a short run. No pass threshold until detailed design sets one. |
| Planned | Generation throughput smoke | Report images/sec, sampling steps, GPU count, elapsed time, ETA, and total generated images for a temporary subset. No pass threshold until detailed design sets one. |
| Planned | Full generation benchmark | Generate all 2,000 requested images and record elapsed time, GPU count, sampling steps, and output validation result. Threshold should fit comfortably inside the under-8-hour full train+generate budget. |
| Planned | Multi-GPU scaling benchmark | Compare 1 GPU vs `torchrun` up to 4 GPUs for training smoke and generation smoke; pass if outputs are complete and no duplicate/missing filenames occur. Speedup threshold Unknown. |
| Planned | Local metric benchmark | Report FID and CLIP-T for the generated 2,000-image set when the scorer adapter is ready. Initial quality target from proposal: FID `<= 90.0142`, CLIP-T `>= 0.2170`. |
| Planned | Full training wall-clock target | Target under 8 hours on 4 RTX 3090 GPUs using `torchrun`; detailed design should set checkpoint and early-stop cadence. |

Performance benchmark results must be labeled benchmark-only unless they are part of a pass/fail gate.

## Evaluator or Grading Commands

No evaluator or grading commands were run while creating this plan.

| Status | Command | Notes |
| --- | --- | --- |
| Known, not run | `python3 score.py --input_dir $input --output_dir $output --config config.json` from `scoring_program/metadata` | Requires evaluator-provided `$input` and `$output`; writes `scores.json`; requires CUDA and complete scorer input. |
| Known, not run | `python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose` from `scoring_manual.txt` | Documented command uses `--score`, but checked-in `score.py` defines `--scores`; not runnable as-is. |
| Known, not run | `python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --scores fid clip_t clip_i --verbose` from `doc/quality-gates.md` | Script-compatible command, but current checkout lacks `input/ref/test.json`, `input/ref/test/`, and result images; would write `scoring_program/scores.json`. |
| Unknown | Exact 2,000-image local scoring adapter command | HLD requires a separate wrapper. FID can be scored without validation images using `test_mu.npy`/`test_sigma.npy`; CLIP-T can be scored without validation images using generated images and prompts. Detailed design must define the command. |
| Missing | Training command | No `scripts/train.py` exists yet. |
| Missing | Generation command | No `scripts/generate.py` exists yet. |
| Missing | Output validation command | No `scripts/validate_outputs.py` exists yet. |
| Missing | Unit test command | `doc/quality-gates.md` reports no discovered unit test command; `AGENTS.md` recommends `python -m unittest discover` after tests exist. |
| Missing | Build/syntax command for future scripts | `AGENTS.md` recommends `python -m compileall scoring_program`; future implementation should expand this to new `scripts/` modules after they exist. |
| Missing | Lint/format/type-check commands | `ruff` and `mypy` commands are recommended only after tools/config/dependencies are installed. |

## Regression Tests

Planned regression checks:

- Scorer argument mismatch: documented `--score` must not be copied into README or scripts when the checked-in parser requires `--scores`.
- 3,000-vs-2,000 mismatch: generated-output validation and development scoring must target the official 2,000 requested files.
- Missing scoring inputs: scoring adapter must fail clearly before FID without reference statistics or before CLIP-T without prompts. It must not require raw validation images.
- Hardcoded `cuda:0`: training and generation must use explicit device selection and must not assume device 0 is free.
- Output overwrite protection: final `generated_images/`, `model.pth`, `checkpoints/`, `runs/`, and score files are not overwritten without explicit opt-in.
- No-pretrained policy: training/generation code must not load OpenCLIP, TorchVision pretrained weights, pretrained VAE/CLIP conditioning, existing checkpoints, high-level generation pipelines, or pretrained-model-generated/labeled extra data. Evaluator-only pretrained weights are allowed only in scoring reports.
- Filename correctness: multi-GPU generation must not drop, duplicate, or reorder filenames in a way that breaks output matching.
- Prompt/category consistency: generated request parsing must preserve multi-word object prompts exactly.
- Submission package completeness: final archive must include required top-level entries and use `HW6_314511048.zip`.

## Manual Verification

- Review `README.md` for reproducible setup, training, generation, validation, scoring, seed, hardware, and extra-data instructions.
- Review `requirements.txt` for pip compatibility and absence of prohibited high-level generation pipeline dependencies.
- Inspect training/generation logs to confirm elapsed time, ETA, loss/progress, device list, seed, and output paths are recorded.
- Inspect a sample grid across animal-object conditions for obvious conditioning failures before spending full Codabench submissions.
- Review extra-data provenance records before enabling external extra data in final training. Default first run uses provided `dataset/` plus class-preserving augmentation.
- Confirm final `generated_images/` visually contains plausible RGB images and not blank/noise-only outputs.
- Confirm local scoring reports state whether CLIP/Inception evaluator weights were cached/downloaded, that raw validation images were not required for CLIP-T, and that evaluator weights were not used by the generator path.
- Confirm Codabench upload result is recorded separately from local development scores.

## Minimum Done Criteria

Before detailed implementation can be considered ready to proceed:

- `doc/test-plan.md` exists with planned verification coverage for every HLD module.
- Exact implementation commands or documented Unknowns are carried into detailed design.
- Scorer limitations are explicit and no unrun evaluator command is presented as verified.

Before implementation can be considered complete:

- Dataset parser tests pass for valid rows, invalid rows, duplicate IDs, unknown labels, prompt consistency, and image existence/dimensions.
- Extra-data validation either passes with complete provenance or external extra data is disabled with an explicit note. Provided `dataset/` plus augmentation is the default.
- Model/diffusion smoke test passes with finite loss and gradients from random initialization.
- No-pretrained-path checks pass for training and generation code.
- Training smoke test writes progress and persistent logs without touching final generated outputs.
- Generation smoke test writes a temporary valid PNG set with deterministic filenames and logs.
- Output validator passes for the final `generated_images/` and fails for controlled bad fixtures.
- Final `generated_images/` contains exactly 2,000 RGB PNG files, all `64x64`, with filenames equal to `dataset/generate.csv`.
- Multi-GPU behavior is tested through `torchrun` for request partitioning and max 4-GPU enforcement, or unsupported hardware conditions are documented.
- Local 2,000-image FID/CLIP-T scoring is run or explicitly blocked with the missing reference-statistics, prompt, evaluator-weight, or adapter reason.
- `README.md`, `requirements.txt`, `scripts/`, `model.pth`, and `generated_images/` exist before packaging.
- Submission packager produces `HW6_314511048.zip` with required contents.
- Known evaluator, test, lint, format, and type-check commands are labeled as run only after they have actually been run.

## Open Questions

- Exact external extra-data source remains open. Recommendation: start with provided `dataset/` plus class-preserving augmentation; add curated external data only after baseline scoring shows a need.
- Exact provenance fields remain open. Minimum recommended fields: source URL/path, retrieval date, license/allowance note, animal, object, label author/method, image hash, and preprocessing notes.
- Exact CLI flags should be defined in detailed design. Defaults should use repository `dataset/`; other flags are delegated to detailed design.
- Test dependency decision: use `pytest` if a test dependency is acceptable; otherwise fall back to `unittest`.
- Multi-GPU training decision: use `torchrun` with native PyTorch distributed execution.
- Reproducibility decision: require deterministic seeds, logged config, exact output filename set, and repeatable smoke outputs on the same device mode. Do not require bitwise identity across single-GPU and multi-GPU modes.
- Scoring adapter decision: implement a separate FID/CLIP-T wrapper. FID uses generated images plus reference statistics; CLIP-T uses generated images plus prompts, so raw validation images are not required.
