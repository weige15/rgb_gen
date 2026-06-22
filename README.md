# rgb_gen

Scratch PyTorch conditional DDPM for HW6 Brainrot Image Generation.

The generator path is implemented from random initialization. Training and generation code do not use pretrained generative weights, pretrained conditioning models, ready-made diffusion pipelines, filtering, or reranking. Pretrained Inception/OpenCLIP weights are used only by the optional local scoring adapter.

## Setup

Use `pip`:

```bash
python -m pip install -r requirements.txt
```

This install command was not run during the latest implementation pass.

## Checks

Available local checks:

```bash
python -m unittest discover
python -m compileall scoring_program scripts
```

Lint, format, and type-check configs are not currently configured.

## Train

Default training input paths are `dataset/train.csv` and `dataset/trainset/`. The default output is `model.pth`; existing output files are refused unless `--overwrite` is passed. Training length is controlled only by `--max_steps`; `--epochs` is not supported.

```bash
python scripts/train.py \
  --train_csv dataset/train.csv \
  --image_dir dataset/trainset \
  --output_model model.pth \
  --run_dir runs/train \
  --seed 1234 \
  --max_steps 100000
```

CPU smoke runs are supported for code-path checks only:

```bash
python scripts/train.py \
  --train_csv dataset/train.csv \
  --image_dir dataset/trainset \
  --output_model /tmp/brainrot-smoke/model.pth \
  --run_dir /tmp/brainrot-smoke/run \
  --max_steps 1 \
  --batch_size 2 \
  --train_timesteps 4 \
  --sampling_steps 4 \
  --base_channels 4 \
  --channel_multipliers 1 \
  --embedding_dim 16 \
  --cpu_smoke
```

For multi-GPU training, use `torchrun` with at most 4 selected CUDA devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py \
  --train_csv dataset/train.csv \
  --image_dir dataset/trainset \
  --output_model model.pth \
  --run_dir runs/train \
  --max_steps 100000
```

Extra external data is disabled by default. If used, pass `--extra_manifest`; the manifest must include provenance, labels, hashes, and must not contain pretrained-model-generated images or labels.

## Generate

Generate images from a trained checkpoint:

```bash
python scripts/generate.py \
  --model model.pth \
  --generate_csv dataset/generate.csv \
  --output_dir generated_images \
  --seed 1234
```

The generator writes RGB PNG files named exactly from `dataset/generate.csv`. Existing non-empty output directories are refused unless `--overwrite` is passed. `--limit` is intended for temporary smoke runs, not final submission generation.

## Validate

Validate the final output contract:

```bash
python scripts/validate_outputs.py \
  --generate_csv dataset/generate.csv \
  --image_dir generated_images \
  --expected_count 2000 \
  --strict_prompt
```

This checks filename set equality, count, PNG format, RGB mode, and 64x64 dimensions.

## Score Locally

Optional development scoring for the 2,000 generated images:

```bash
python scripts/score_2000.py \
  --image_dir generated_images \
  --generate_csv dataset/generate.csv \
  --ref_mu scoring_program/input/ref/test_mu.npy \
  --ref_sigma scoring_program/input/ref/test_sigma.npy \
  --scores fid clip_t \
  --output scores_2000.json
```

Local scoring may require cached or downloadable evaluator weights. FID uses Inception features and the provided reference statistics. CLIP-T uses generated images plus prompts from `dataset/generate.csv`; raw validation images are not required. Codabench remains authoritative.

The original `scoring_program/score.py` is left unchanged. Its parser uses `--scores`, while `scoring_manual.txt` shows `--score`.

## Package

Package the E3 archive after `generated_images/`, `model.pth`, `README.md`, and `requirements.txt` exist and validation passes:

```bash
python scripts/package_submission.py \
  --student_id 314511048 \
  --generated_dir generated_images \
  --scripts_dir scripts \
  --model model.pth \
  --readme README.md \
  --requirements requirements.txt \
  --generate_csv dataset/generate.csv \
  --output HW6_314511048.zip
```

The packager uses `zipfile`, requires the archive name `HW6_314511048.zip`, and excludes development-only `checkpoints/`, `runs/`, bytecode caches, and score JSON files from `scripts/`.

## Current Status

Implemented modules live under `scripts/` with `unittest` coverage under `tests/`. The latest implementation pass did not run full training, full generation, real local scoring, dependency installation, or final packaging against real artifacts.
