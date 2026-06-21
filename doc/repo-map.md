# Repository Map

## Repository Summary

This repository is `rgb_gen` per `README.md`. It is an HW6 Brainrot conditional image-generation assignment workspace with:

- Training metadata and images under `dataset/`.
- A local Python scoring program under `scoring_program/`.
- Assignment/scoring documentation in `HW6-Brainrot Image Generation.pdf`, `scoring_manual.txt`, and `doc/problem-brief.md`.

The repository does not currently contain a generation model implementation, training scripts, generated output images, model weights, or Python dependency manifest.

## Directory Structure

- `README.md` - repository title only.
- `AGENTS.md` - local agent/repository guidance. It is currently untracked.
- `.gitignore` - ignores Python bytecode, model weights, generated images, checkpoints, run logs, and root `scores.json`.
- `dataset/`
  - `train.csv` - 4,800 lines including header; 4,799 training records.
  - `generate.csv` - 2,001 lines including header; 2,000 generation requests.
  - `trainset/` - 4,799 PNG training images.
- `scoring_program/`
  - `score.py` - local scorer implementation.
  - `metadata` - platform-style scoring command description.
  - `scores.json` - existing scorer output containing an FID value.
  - `input/ref/` - scorer reference assets: `config.json`, `test_mu.npy`, `test_sigma.npy`.
  - `input/res/` - scorer result-image directory; currently contains 0 PNG files.
  - `.git/` - empty directory observed; active git root still resolves to the repository root.
- `doc/`
  - `problem-brief.md` - source-grounded assignment brief. `doc/` is currently untracked.
- `.agents/` and `.codex/` - directories observed, with no files found at max depth 3.

## Main Source Files

- `scoring_program/score.py`
  - Python CLI for image-generation evaluation.
  - Computes FID using Inception v3 features and reference mean/covariance arrays.
  - Computes CLIP image-text (`clip_t`) and image-image (`clip_i`) scores with OpenCLIP.
  - Checks PNG count and optional image size.
  - Loads optional JSON config from `${input_dir}/ref/<config>`.
  - Writes `scores.json` to `--output_dir`.
  - Uses `torch.device("cuda:0")` directly, so the script expects CUDA unless modified.

No training, generation, model, dataset loading, or submission packaging source files were found.

## Existing Tests

No test directory or runnable test suite was found.

Files with `test` in their names are scorer reference artifacts:

- `scoring_program/input/ref/test_mu.npy`
- `scoring_program/input/ref/test_sigma.npy`

The scorer documentation references `scoring_program/input/ref/test/` and `scoring_program/input/ref/test.json`, but those files/directories are not present.

## Build System

No build or package metadata files were found at max depth 3:

- No `requirements*.txt`.
- No `pyproject.toml`.
- No `setup.py`.
- No `package.json`.
- No `Makefile`.
- No `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, or `settings.gradle`.

The only executable workflow confirmed from files is the Python scoring script.

## Runtime or CLI Entry Points

- Confirmed scorer entry point:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --scores fid clip_t clip_i --verbose
```

- Confirmed metadata command:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

`scoring_manual.txt` shows `--score fid clip_t clip_i`, while `score.py` defines `--scores`. The script form is the confirmed implementation.

No application, training, or generation CLI entry point was found.

## Data and Assets

- `dataset/train.csv`
  - Columns: `id,animal,object`.
  - Contains 4,799 records after the header.
  - Contains 10 animal categories and 10 object categories.
- `dataset/generate.csv`
  - Columns: `id,animal,object,prompt`.
  - Contains 2,000 records after the header.
  - Contains 10 animal categories, 10 object categories, and 100 animal-object pairs.
  - Prompt format in observed rows: `a {animal} and a {object}`.
- `dataset/trainset/`
  - Contains 4,799 PNG files.
  - Sample file `dataset/trainset/000001.png` is 64x64 RGB PNG.
- `HW6-Brainrot Image Generation.pdf`
  - 5-page assignment PDF.
  - Describes the assignment objective, constraints, grading metrics, and submission requirements.
- `scoring_program/input/ref/test_mu.npy`
  - NumPy reference mean array for FID.
- `scoring_program/input/ref/test_sigma.npy`
  - NumPy reference covariance array for FID.
- `scoring_program/input/ref/config.json`
  - Scorer config: image size 64, number of images 3000, scores `fid`, `clip_t`, `clip_i`, batch size 32, 4 workers, verbose true.
- `scoring_program/input/res/`
  - Present but contains 0 PNG files.

No `generated_images/`, `checkpoints/`, `runs/`, `model.pth`, or `scripts/` directory was found.

## Existing Documentation

- `README.md` - contains only `# rgb_gen`.
- `AGENTS.md` - documents project context, command rules, known scoring facts, and missing build/test commands.
- `HW6-Brainrot Image Generation.pdf` - assignment specification in Chinese.
- `doc/problem-brief.md` - detailed assignment brief derived from the PDF and scoring manual.
- `scoring_manual.txt` - local scorer setup and usage instructions.
- `scoring_program/metadata` - Codabench-style scoring command metadata.

## Detected Dependencies

Detected from `scoring_program/score.py` imports:

- Python standard library: `argparse`, `glob`, `json`, `os`, `collections.defaultdict`, `typing`.
- External Python packages: `numpy`, `open_clip`, `torch`, `Pillow` via `PIL`, `torchvision`, `tqdm`, `scipy`.

No dependency versions are specified in the repository because no dependency manifest was found.

## Important Scripts

- `scoring_program/score.py`
  - Main local evaluation script.
  - CLI options include `--input_dir`, `--output_dir`, `--image_size`, `--num_images`, `--scores`, `--ref_mu`, `--ref_sigma`, `--test_json`, `--test_image_root`, `--model_name`, `--pretrained`, `--batch_size`, `--num_workers`, `--verbose`, and `--config`.

No training, generation, download, packaging, lint, format, type-check, or test scripts were found.

## Current Git State

Observed git facts:

- Repository root: `/home/kuotzuwei15/GenAI/rgb`.
- Branch: `main`.
- Upstream shown by `git status --short --branch`: `origin/main`.
- Worktree list: `/home/kuotzuwei15/GenAI/rgb  b4c0706 [main]`.
- Before creating this map, `git status --short --branch` showed:
  - `?? AGENTS.md`
  - `?? doc/`

Creating this repository map adds `doc/repo-map.md` inside the already-untracked `doc/` directory.

## Missing or Ambiguous Areas

- No implemented image-generation model was found.
- No training or generation scripts were found.
- No `requirements.txt` or other dependency manifest was found.
- No runnable test suite was found.
- No build, lint, format, or type-check commands were found.
- No generated image output directory was found.
- No model weight file was found.
- The scorer config expects `num_images: 3000`, while assignment generation data contains 2,000 requested images.
- `scoring_manual.txt` uses `--score` in the example command, but `score.py` defines `--scores`.
- `scoring_program/input/ref/test/` and `scoring_program/input/ref/test.json` are referenced by docs/script defaults but are absent.
- `scoring_program/.git/` exists as an empty directory, but `git -C scoring_program rev-parse --show-toplevel` resolves to the repository root.

## Notes for Future Skills

- Use this map as the factual starting point for proposal, design, test-plan, or implementation skills.
- Future implementation work should preserve the assignment constraints from `doc/problem-brief.md`: the main generation model must be trained from scratch, and pretrained generative pipelines/weights are not allowed.
- Before implementation, decide whether local evaluation should target the assignment's 2,000 generated images or the scorer config/manual's 3,000-image reference setup.
- Add a dependency manifest when introducing training/generation code.
- Add minimal runnable checks when introducing non-trivial logic.
