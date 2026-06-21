# Implementation Prompt

## Objective

Implement the HW6 Brainrot conditional image-generation project end to end from the existing planning docs. Build a reproducible, from-scratch PyTorch conditional DDPM that trains on the local Brainrot dataset, generates exactly 2,000 64x64 RGB PNG images named from `dataset/generate.csv`, validates the output contract, optionally scores the 2,000-image set for local FID/CLIP-T development feedback, and packages the final E3 archive.

Do not treat this prompt as permission to skip the docs. This prompt is the entry point; the docs and task files are the source of truth for module contracts and acceptance criteria.

## Inputs to Read First

Read these first, in this order:

- `AGENTS.md`: present. Repository rules, permissions, generated-artifact protections, known scoring facts, and recommended checks.
- `doc/problem-brief.md`: present optional input. Assignment objective, data schema, constraints, deliverables, grading thresholds, and no-pretrained restrictions.
- `doc/repo-map.md`: present optional input. Current repository state, missing implementation files, dataset/scorer facts, and command gaps.
- `doc/quality-gates.md`: present optional input. Discovered commands, missing gates, scorer caveats, and minimum done criteria.
- `doc/proposal.md`: present. Product intent, scratch DDPM approach, constraints, milestones, validation plan, and risks.
- `doc/high-level-design.md`: present. Module boundaries, data flow, dependency direction, progress/multi-GPU requirements, and scoring strategy.
- `doc/test-plan.md`: present. Module tests, integration tests, golden cases, edge cases, regression checks, performance benchmarks, and evaluator commands.
- `doc/detailed-design.md`: present. Exact shared contracts, planned files, CLI shapes, module interfaces, algorithms, failure handling, and quality gates.
- `doc/tasks/progress.md`: present. Maintain this file while implementing.
- `doc/tasks/data-and-condition-registry.md`
- `doc/tasks/extra-data-intake-and-provenance.md`
- `doc/tasks/scratch-conditional-denoiser.md`
- `doc/tasks/diffusion-process-and-sampler.md`
- `doc/tasks/training-orchestrator.md`
- `doc/tasks/generation-orchestrator.md`
- `doc/tasks/output-validator.md`
- `doc/tasks/2-000-image-scoring-adapter.md`
- `doc/tasks/submission-packager.md`
- `doc/tasks/documentation-and-dependency-manifest.md`

Inspect these current implementation files before coding:

- `README.md`
- `.gitignore`
- `dataset/train.csv`
- `dataset/generate.csv`
- representative images in `dataset/trainset/`
- `scoring_manual.txt`
- `scoring_program/score.py`
- `scoring_program/metadata`
- `scoring_program/input/ref/config.json`
- `scoring_program/input/ref/test_mu.npy`
- `scoring_program/input/ref/test_sigma.npy`

## Current Implementation

Repository facts at prompt creation time:

- Repository root: `/home/kuotzuwei15/GenAI/rgb`.
- Branch: `main`, upstream `origin/main`.
- Worktree status included untracked `AGENTS.md` and `doc/`.
- Project name in `README.md`: `rgb_gen`; README currently contains only `# rgb_gen`.
- `.gitignore` ignores Python bytecode, `*.pth`, `*.pt`, `generated_images/`, `checkpoints/`, `runs/`, and root `scores.json`.
- Dataset exists:
  - `dataset/train.csv`: 4,799 data rows, columns `id,animal,object`.
  - `dataset/generate.csv`: 2,000 data rows, columns `id,animal,object,prompt`.
  - `dataset/trainset/`: 4,799 PNG training images.
- No `scripts/` directory exists yet.
- No `tests/` directory exists yet.
- No `requirements.txt`, `pyproject.toml`, `setup.py`, lockfile, Makefile, or CI config exists.
- No `generated_images/`, `checkpoints/`, `runs/`, or `model.pth` exists.
- Existing executable code is `scoring_program/score.py`.
- `scoring_program/score.py` imports `numpy`, `open_clip`, `torch`, `PIL`, `torchvision`, `tqdm`, and `scipy`; it computes FID and CLIP scores and writes `scores.json`.
- `scoring_program/score.py` hardcodes `torch.device("cuda:0")` in its CLI path and defines `--scores`, while `scoring_manual.txt` documents `--score`.
- Local scorer assets present: `scoring_program/input/ref/config.json`, `test_mu.npy`, and `test_sigma.npy`.
- Local scorer assets missing for the stock scorer CLIP path: `scoring_program/input/ref/test.json` and `scoring_program/input/ref/test/`.
- `scoring_program/input/ref/config.json` expects `num_images: 3000`, but the assignment generation target is 2,000 images.
- No build, test, lint, format, type-check, static-analysis, smoke-test, training, generation, validation, packaging, or complete local evaluator command is currently runnable as a verified project workflow.

## Hard Constraints

- The main generator must be trained from scratch.
- Do not use pretrained generative weights, pretrained UNets, pretrained Transformers, pretrained diffusion models, ready-made generation pipelines, or existing checkpoints in training or generation.
- Do not copy public homework solutions or public repository code.
- Do not use pretrained or existing models for generator conditioning, training targets, filtering, reranking, or final-image generation.
- Evaluator-only pretrained Inception/OpenCLIP weights may be used only inside local scoring code and reports; training and generation modules must not import the scoring adapter.
- Do not use hidden test images or hidden test prompts in training, validation, or generation.
- Generate exactly 2,000 PNG files for `dataset/generate.csv`, not 3,000.
- Every final generated image must be RGB PNG, 64x64, and named exactly from `dataset/generate.csv`.
- Do not modify the large tracked dataset or scorer reference assets unless the user explicitly asks.
- Do not overwrite `generated_images/`, `model.pth`, `checkpoints/`, `runs/`, `scores.json`, `scoring_program/scores.json`, or archives without explicit overwrite flags and user awareness.
- Do not hardcode `cuda:0` in new training or generation code.
- Enforce at most 4 GPUs for project training/generation.
- Keep single-GPU and CPU smoke-check paths where practical, but do not promise CPU full training.
- Use `pip` as the package manager. Add `requirements.txt` before documenting install commands as runnable.
- Prefer Python standard library where sufficient. Do not add high-level generation pipeline dependencies such as `diffusers`.
- Keep `scoring_program/score.py` unchanged unless a later explicit scoring task asks to modify it.
- Use `unittest` by default unless you deliberately add and document a test dependency.
- Preserve user/other-agent edits; do not revert unrelated changes.

## Non-Goals

- Do not pursue full training quality before the code path, smoke tests, validator, and reproducibility docs exist.
- Do not require bitwise-identical outputs across CPU, single-GPU, and multi-GPU modes.
- Do not make local scorer results authoritative over Codabench.
- Do not require external extra data for the first working baseline.
- Do not create generated final artifacts during unit tests.
- Do not commit, push, branch, rebase, merge, or stash unless the user explicitly asks.

## Execution Model

- Start with read-only discovery: `pwd`, `git status --short --branch`, `git branch --show-current`, `git worktree list`, `rg --files`, relevant doc reads, and current file inspection.
- Read all inputs listed above before implementation.
- Keep `doc/tasks/progress.md` current. Mark a module as started before writing code, mark completed only after its task file done criteria and checks pass, and add timestamped notes for blocked or verified work.
- Implement one module or small workstream at a time. Do not start broad parallel edits in the same files.
- Use subagents only for independent workstreams with disjoint write scopes. The main agent owns integration, final quality gates, shared contracts, and conflict resolution.
- Keep write scopes disjoint. If two workstreams need `scripts/brainrot_data.py`, do the base data registry first, then layer extra-data changes in a separate pass.
- After each module, run the smallest relevant checks from that task file and summarize the result in progress.
- At the end, run the full available quality gates and summarize command output evidence.
- If a check cannot be run because dependencies, CUDA, evaluator files, network, or generated artifacts are missing, record the exact blocker in `doc/tasks/progress.md` and the final response.
- Stop and ask only when truly blocked by missing requirements, destructive choices, credentials, external services, or conflicting docs that cannot be resolved conservatively.

## Module Workstreams

1. Data and Condition Registry
   - Task: `doc/tasks/data-and-condition-registry.md`
   - Owns: `scripts/brainrot_data.py`, `tests/test_brainrot_data.py`, test fixtures under `tests/fixtures/`.
   - Expected public pieces: assignment vocabularies, `Condition`, `TrainRecord`, `GenerationRequest`, `load_train_records`, `load_generation_requests`, `build_condition`, `validate_prompt`, `BrainrotDataset`.
   - Verification: `python -m unittest tests.test_brainrot_data`; `python -m compileall scripts`.

2. Extra Data Intake and Provenance
   - Task: `doc/tasks/extra-data-intake-and-provenance.md`
   - Owns: extra-data helpers in `scripts/brainrot_data.py` or `scripts/extra_data.py`, `tests/test_extra_data.py`.
   - Boundary: depends on data registry; defaults to no external data; performs no downloads.
   - Verification: `python -m unittest tests.test_extra_data`; `python -m unittest tests.test_brainrot_data`.

3. Scratch Conditional Denoiser
   - Task: `doc/tasks/scratch-conditional-denoiser.md`
   - Owns: `scripts/model.py`, `tests/test_model.py`.
   - Boundary: no filesystem reads, no pretrained weights, no scoring imports, no CUDA init at import.
   - Verification: `python -m unittest tests.test_model`; `python -m compileall scripts`.

4. Diffusion Process and Sampler
   - Task: `doc/tasks/diffusion-process-and-sampler.md`
   - Owns: `scripts/diffusion.py`, `tests/test_diffusion.py`.
   - Boundary: owns schedules, `q_sample`, loss, DDPM sampling, CFG mixing, and pixel conversion helper; does not own CLI.
   - Verification: `python -m unittest tests.test_diffusion`; `python -m unittest tests.test_model`.

5. Training Orchestrator
   - Task: `doc/tasks/training-orchestrator.md`
   - Owns: `scripts/train.py`, `tests/test_train.py`, shared runtime helpers only if needed.
   - Boundary: writes only temp outputs in tests; final `model.pth`, `checkpoints/`, and `runs/` are explicit CLI outputs guarded by overwrite checks.
   - Verification: `python -m unittest tests.test_train`; `python -m unittest tests.test_brainrot_data tests.test_model tests.test_diffusion`; `python -m compileall scripts`.

6. Generation Orchestrator
   - Task: `doc/tasks/generation-orchestrator.md`
   - Owns: `scripts/generate.py`, `tests/test_generate.py`.
   - Boundary: temporary output dirs in tests; final `generated_images/` only when explicitly requested.
   - Verification: `python -m unittest tests.test_generate`; dependency tests; validator smoke after validator exists.

7. Output Validator
   - Task: `doc/tasks/output-validator.md`
   - Owns: `scripts/validate_outputs.py`, `tests/test_validate_outputs.py`.
   - Boundary: read-only inspection of generated-image dirs; no image modification.
   - Verification: `python -m unittest tests.test_validate_outputs`; `python -m compileall scripts`.

8. 2,000-Image Scoring Adapter
   - Task: `doc/tasks/2-000-image-scoring-adapter.md`
   - Owns: `scripts/score_2000.py`, `tests/test_score_2000.py`.
   - Boundary: development-only evaluator; may load pretrained evaluator weights only when scoring actually runs; never imported by training/generation; do not modify `scoring_program/score.py`.
   - Verification: `python -m unittest tests.test_score_2000`; import isolation checks.

9. Submission Packager
   - Task: `doc/tasks/submission-packager.md`
   - Owns: `scripts/package_submission.py`, `tests/test_package_submission.py`.
   - Boundary: uses Output Validator; packages temp fixtures in tests; final archive only by explicit CLI run.
   - Verification: `python -m unittest tests.test_package_submission`; `python -m unittest tests.test_validate_outputs`.

10. Documentation and Dependency Manifest
   - Task: `doc/tasks/documentation-and-dependency-manifest.md`
   - Owns: `README.md`, `requirements.txt`; may update `AGENTS.md` only for newly verified workflows.
   - Boundary: document only verified commands as run; clearly label planned or blocked commands.
   - Verification: `python -m pip install -r requirements.txt` only after dependency installation is approved; `python -m unittest discover`; `python -m compileall scoring_program scripts`; manual README review.

## Subagent Plan

Use subagents only after the main agent has created or confirmed shared contracts in `scripts/brainrot_data.py`, `scripts/model.py`, and `scripts/diffusion.py`.

Good subagent candidates with disjoint writes:

- Data subagent: `scripts/brainrot_data.py`, `tests/test_brainrot_data.py`, fixtures. Main agent reviews shared dataclasses and category contract before other work.
- Model subagent: `scripts/model.py`, `tests/test_model.py`. No data-file writes except imports of condition constants if needed.
- Diffusion subagent: `scripts/diffusion.py`, `tests/test_diffusion.py`. May depend on `scripts.model` interface but should not edit it.
- Validator/packaging subagent: `scripts/validate_outputs.py`, `scripts/package_submission.py`, `tests/test_validate_outputs.py`, `tests/test_package_submission.py`.
- Scoring subagent: `scripts/score_2000.py`, `tests/test_score_2000.py`. Must not edit training/generation or `scoring_program/score.py`.
- Docs subagent: `README.md`, `requirements.txt`, documentation review. Should wait until CLI flags stabilize.

Keep these main-agent owned or integration-only:

- `scripts/train.py`
- `scripts/generate.py`
- shared runtime helpers used by both training and generation
- `doc/tasks/progress.md`
- final integration fixes across modules
- final full quality-gate run

Subagent merge rules:

- Each subagent must report changed files, tests run, and residual risks.
- Main agent re-reads changes before integrating dependents.
- If two subagents need the same file, stop parallel work on that file and serialize edits.
- Never accept subagent changes that weaken no-pretrained, output-contract, overwrite, or device-selection constraints.

## Implementation Order

1. Repository setup for implementation
   - Create `scripts/` and `tests/` only when implementing code.
   - Add minimal package markers only if imports need them.
   - Update `doc/tasks/progress.md` with a started checkpoint.
   - Local check: `python -m compileall scoring_program` remains the only existing syntax check before scripts exist.

2. Data and Condition Registry
   - Implement `scripts/brainrot_data.py` and `tests/test_brainrot_data.py`.
   - Check: `python -m unittest tests.test_brainrot_data`.
   - Check: `python -m compileall scripts`.

3. Extra Data Intake and Provenance
   - Add no-manifest and manifest-validation paths.
   - Check: `python -m unittest tests.test_extra_data tests.test_brainrot_data`.

4. Scratch Conditional Denoiser
   - Implement `scripts/model.py` and tensor-only tests.
   - Check: `python -m unittest tests.test_model`.

5. Diffusion Process and Sampler
   - Implement `scripts/diffusion.py` and formula-oracle tests.
   - Check: `python -m unittest tests.test_diffusion tests.test_model`.

6. Output Validator
   - Implement validation CLI and fixtures.
   - Check: `python -m unittest tests.test_validate_outputs`.

7. Training Orchestrator
   - Implement safe CLI, one-step smoke, DDP config validation, checkpoint contract.
   - Check: `python -m unittest tests.test_train tests.test_brainrot_data tests.test_model tests.test_diffusion`.

8. Generation Orchestrator
   - Implement checkpoint load, partitioning, temp-output smoke generation.
   - Check: `python -m unittest tests.test_generate tests.test_validate_outputs`.

9. 2,000-Image Scoring Adapter
   - Implement local scoring wrapper and missing-fixture tests.
   - Check: `python -m unittest tests.test_score_2000`.
   - Do not run real FID/CLIP-T unless generated outputs and evaluator weights are available and the user approves any required downloads or score-file writes.

10. Submission Packager
   - Implement archive assembly over validated temp fixtures.
   - Check: `python -m unittest tests.test_package_submission tests.test_validate_outputs`.

11. Documentation and Dependency Manifest
   - Add `requirements.txt`.
   - Update `README.md` with verified setup, train, generate, validate, score, package, seed, hardware, overwrite, and no-pretrained notes.
   - Do not claim any command or metric was run unless it was run in this session.

12. Final quality gates
   - Run all configured/available checks listed below.
   - Update `doc/tasks/progress.md` with pass/fail/blocker evidence.

## Testing and Quality Gates

Use actual available and planned commands from `doc/quality-gates.md`, `AGENTS.md`, and task files.

Existing verified state before implementation:

- No discovered build command.
- No discovered unit test command.
- No discovered lint command.
- No discovered format command.
- No discovered type-check command.
- No discovered static-analysis command.
- No discovered smoke-test command.
- No training/generation/validation/packaging commands exist yet.
- Existing scorer commands are not runnable as-is because local reference `test.json`, reference test images, and generated result images are missing; scoring also requires CUDA and may require evaluator weights.

Required checks after implementation creates the relevant files:

```bash
python -m unittest tests.test_brainrot_data
python -m unittest tests.test_extra_data
python -m unittest tests.test_model
python -m unittest tests.test_diffusion
python -m unittest tests.test_validate_outputs
python -m unittest tests.test_train
python -m unittest tests.test_generate
python -m unittest tests.test_score_2000
python -m unittest tests.test_package_submission
python -m unittest discover
python -m compileall scoring_program scripts
```

Dependency install check, only after `requirements.txt` exists and dependency installation is approved:

```bash
python -m pip install -r requirements.txt
```

Optional lint/format/type checks, only after tools/config are added:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy scripts scoring_program
```

Planned project CLI checks after implementation:

```bash
python scripts/validate_outputs.py --generate_csv dataset/generate.csv --image_dir generated_images --expected_count 2000
python scripts/score_2000.py --image_dir generated_images --generate_csv dataset/generate.csv --ref_mu scoring_program/input/ref/test_mu.npy --ref_sigma scoring_program/input/ref/test_sigma.npy --scores fid clip_t
python scripts/package_submission.py --student_id 314511048 --output HW6_314511048.zip
```

Training and generation commands are planned and potentially long-running. Run smoke versions first using temp outputs and explicit small limits. Run full training/generation only when the user has approved runtime, devices, overwrite behavior, and generated-artifact writes:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --output_model model.pth
python scripts/generate.py --model model.pth --generate_csv dataset/generate.csv --output_dir generated_images
```

Known stock scorer commands, for reference only:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --scores fid clip_t clip_i --verbose
python3 score.py --input_dir $input --output_dir $output --config config.json
```

Do not copy the documented `--score` form into implementation docs as the checked-in parser uses `--scores`.

## Progress Tracking

Maintain `doc/tasks/progress.md` throughout implementation.

Requirements:

- Before starting a module, add a timestamped checkpoint line under that module or an implementation notes section.
- Mark a module checkbox complete only when its task file `Done When` criteria are satisfied and relevant checks have passed or are explicitly blocked with reasons.
- Record command summaries: command, exit status, key output, and blocker if any.
- Record generated-artifact actions, including any approved writes to `model.pth`, `generated_images/`, `checkpoints/`, `runs/`, score files, or archives.
- Keep full-project gate checkboxes unchecked until the corresponding full gate has actually passed.
- Do not delete existing progress history unless it is plainly obsolete and explain the rewrite.

## Commit or Checkpoint Strategy

Do not commit unless the user explicitly requests commits.

If commits are requested, make logical commits in dependency order:

1. Data and validation foundations.
2. Model and diffusion core.
3. Training and generation CLIs.
4. Scoring and packaging.
5. Docs and dependency manifest.

Without commits, keep the final diff organized by module and report changed files grouped by workstream. Before finishing, run `git status --short --branch` and summarize the remaining untracked/modified files. Never use destructive git commands such as `git reset --hard`, `git clean -fd`, `git restore .`, or force pushes unless the user explicitly asks for that exact action.

## Acceptance Criteria

Implementation is complete only when:

- Every task in `doc/tasks/*.md` is completed or explicitly marked blocked with a concrete reason.
- `doc/tasks/progress.md` reflects completed, blocked, and verified work.
- `scripts/brainrot_data.py` implements CSV parsing, category mapping, prompt validation, image checks, and dataset loading.
- Optional extra-data intake defaults to no external data and fails closed on missing provenance or prohibited pretrained-generated/labeled data.
- `scripts/model.py` implements a scratch conditional denoiser from random initialization.
- `scripts/diffusion.py` implements schedule construction, `q_sample`, training loss, DDPM sampling, CFG support, and image conversion.
- `scripts/train.py` supports safe smoke training, random initialization, progress/logging, checkpoint contract, explicit devices, and max 4-GPU enforcement.
- `scripts/generate.py` loads `model.pth`, preserves request filenames, writes RGB 64x64 PNGs, supports temp smoke generation, and protects final outputs from accidental overwrite.
- `scripts/validate_outputs.py` verifies exact count, filename set, PNG format, RGB mode, and 64x64 size.
- `scripts/score_2000.py` scores only the 2,000 generated submission images for local FID/CLIP-T and is isolated from training/generation.
- `scripts/package_submission.py` creates `HW6_314511048.zip` only from validated required artifacts.
- `README.md` documents setup, training, generation, validation, optional scoring, packaging, seeds, hardware assumptions, overwrite behavior, no-pretrained constraints, and extra-data policy.
- `requirements.txt` exists and is pip-compatible.
- Tests are added or updated for all modules.
- `python -m unittest discover` passes.
- `python -m compileall scoring_program scripts` passes.
- Lint, format, type-check, and static-analysis gates pass if configured; otherwise their absence is recorded.
- Evaluator or benchmark passes if configured and runnable; otherwise the exact blocker is recorded.
- Final generated outputs, if produced, pass `python scripts/validate_outputs.py --generate_csv dataset/generate.csv --image_dir generated_images --expected_count 2000`.
- No unrelated files are changed.
- No hidden test data, pretrained generator weights, ready-made generation pipeline, or copied solution code is introduced.

## Uncertainty Protocol

Make conservative, documented assumptions when safe:

- Use repository `dataset/` as the default training source.
- Use `unittest` unless adding a test dependency is clearly worthwhile and documented.
- Keep external extra data disabled unless a valid manifest is provided.
- Keep local scoring separate from generation and label it development-only.
- Prefer no overwrite by default.

Ask the user before proceeding when blocked by:

- Missing or conflicting assignment requirements that affect output format, model restrictions, or scoring.
- Any destructive action or overwrite of final artifacts.
- Long-running full training, full generation, or local scoring that writes score files.
- Dependency installation, network downloads, or first-time evaluator-weight downloads.
- Credentials, cloud links, Codabench upload actions, or external services.
- A quality gate that cannot be run and has no reasonable local substitute.

When blocked, state the exact file, command, missing artifact, or conflicting requirement.

## Final Response Requirements

At the end of the implementation session, respond concisely with:

- Implementation summary grouped by workstream.
- Changed files grouped by module.
- Tests and quality gates run, with command output summaries and exit status.
- Generated artifacts created, if any, with paths and overwrite approvals.
- Known limitations or blocked gates.
- Any required follow-up, such as full training, full generation, Codabench scoring, dependency installation, or evaluator-weight download.

Do not claim FID, CLIP-T, full generation success, full training success, lint/type-check success, or package correctness unless the relevant command was actually run and passed.
