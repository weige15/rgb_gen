# Quality Gates

## Environment Summary

- Repository root: `/home/kuotzuwei15/GenAI/rgb`.
- Current branch at discovery time: `main`.
- Worktree status at discovery time: `## main...origin/main` with untracked `AGENTS.md` and `doc/`.
- Project name from `README.md`: `rgb_gen`.
- Primary language: Python.
- Package manager: Unknown. No `requirements.txt`, `pyproject.toml`, `uv.lock`, `package.json`, lockfile, `Makefile`, `justfile`, CI workflow, or comparable task runner file was discovered.
- Assignment source: `HW6-Brainrot Image Generation.pdf`.
- Local scoring instructions source: `scoring_manual.txt`.
- Existing problem summary source: `doc/problem-brief.md`.
- Main evaluator script: `scoring_program/score.py`.
- Dataset facts verified locally:
  - `dataset/train.csv`: 4,799 data rows plus header.
  - `dataset/generate.csv`: 2,000 data rows plus header.
  - `dataset/trainset/`: 4,799 PNG files.
- Local scorer fixture status:
  - Present: `scoring_program/input/ref/config.json`, `test_mu.npy`, `test_sigma.npy`, and `scoring_program/input/res/`.
  - Missing for full CLIP scoring: `scoring_program/input/ref/test.json` and `scoring_program/input/ref/test/`.
  - `scoring_program/input/res/` currently contains 0 PNG files.
- `scoring_program/score.py` uses `torch.device("cuda:0")`; scoring requires CUDA unless the script is modified.
- `scoring_program/score.py` loads TorchVision Inception weights and OpenCLIP `ViT-B-32-quickgelu` pretrained weights, so scoring may require cached model weights or network access on first use.

## Build Commands

- Missing: No build command was discovered.

## Unit Test Commands

- Missing: No unit test command was discovered.

## Integration Test Commands

- Missing: No integration test command was discovered.

## Lint Commands

- Missing: No lint command was discovered.

## Format Commands

- Missing: No format command was discovered.

## Type-Check Commands

- Missing: No type-check command was discovered.

## Static Analysis Commands

- Missing: No static analysis command was discovered.

## Benchmark or Evaluator Commands

- Discovered: Platform-style evaluator command from `scoring_program/metadata`.
  - Working directory: `scoring_program`
  - Command:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

  - Notes: Requires evaluator-provided `$input` and `$output`; writes `scores.json` under the output directory; requires CUDA; may require cached or downloadable model weights; uses config located under `${input_dir}/ref/config.json`.

- Discovered: Local scoring command exactly as documented in `scoring_manual.txt`.
  - Working directory: `scoring_program`
  - Command:

```bash
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

  - Notes: Requires `./input/ref/test.json`, `./input/ref/test/`, reference statistics, generated PNGs in `./input/res`, CUDA, and model weights. Writes `./scores.json`. The current `score.py` parser defines `--scores`, not `--score`, so this exact documented command appears inconsistent with the checked-in script.

- Inferred: Script-compatible local scoring command derived from `scoring_manual.txt` and the `score.py` argument parser.
  - Working directory: `scoring_program`
  - Command:

```bash
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --scores fid clip_t clip_i --verbose
```

  - Notes: Same runtime requirements as the local scoring command above. This command is not runnable as-is in the current checkout because `input/ref/test.json`, `input/ref/test/`, and generated result images are not present. It writes `scoring_program/scores.json`.

## Smoke Test Commands

- Missing: No lightweight smoke test command was discovered.

## Verified Commands

- None. No build, test, lint, format, type-check, static-analysis, benchmark, smoke-test, or evaluator command was run during this discovery.

## Commands Not Run

- `python3 score.py --input_dir $input --output_dir $output --config config.json`
  - Working directory: `scoring_program`
  - Reason not run: Requires evaluator-provided `$input` and `$output`; current checkout does not provide a complete scorer input fixture for this command.

- `python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose`
  - Working directory: `scoring_program`
  - Reason not run: The checked-in parser uses `--scores`, not `--score`; local reference `test.json`, reference test images, and generated result images are missing; running would write `scoring_program/scores.json`.

- `python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --scores fid clip_t clip_i --verbose`
  - Working directory: `scoring_program`
  - Reason not run: Local reference `test.json`, reference test images, and generated result images are missing; CUDA is required; running would write `scoring_program/scores.json`.

## Missing Quality Gates

- Build: No command found.
- Unit tests: No command found.
- Integration tests: No command found.
- Lint: No command found.
- Format check: No command found.
- Type-check: No command found.
- Static analysis: No command found.
- Smoke test: No command found.
- Dependency installation or environment validation: No command found.
- Training command: No command found.
- Generation command: No command found.
- Submission packaging command: No command found.
- Generated image contract validation command: No command found for checking exactly 2,000 PNGs, filenames matching `dataset/generate.csv`, and 64x64 image dimensions.

## Recommended Minimum Done Criteria

- Preserve the assignment constraints from `HW6-Brainrot Image Generation.pdf`: the main conditional image generation model is trained from scratch, no pretrained generative weights or ready-made high-level generation pipelines are used, and the training/generation flow is reproducible.
- Before final Codabench upload, verify the submission output contract from the assignment: `generated_images/` contains exactly 2,000 PNG files, filenames match `dataset/generate.csv`, and every image is 64x64.
- Before E3 submission, verify the archive contains `generated_images/`, `scripts/`, `model.pth`, `README.md`, and `requirements.txt`, and that `README.md` documents the environment, training flow, and generation commands.
- Once the local scorer fixture is complete, run the script-compatible scorer command from the repository root by entering `scoring_program/` and using `--scores fid clip_t clip_i`; record the exit code and resulting `scores.json`.
- Add one small generated-image validation command or script before implementation is considered complete; no such command exists in the repository today.
