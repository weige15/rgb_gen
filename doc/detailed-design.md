# Detailed Design

## Purpose

Define the implementation design for the scratch conditional Brainrot image generator described by `doc/proposal.md`, `doc/high-level-design.md`, and `doc/test-plan.md`.

This document is a design artifact only. It does not implement code, run training, generate images, run the scorer, or modify generated artifacts.

## Source Proposal Summary

The proposal selects a from-scratch pixel-space conditional DDPM in PyTorch for 64x64 RGB Brainrot images. The generator trains only from random initialization, uses the local Brainrot dataset as the first baseline, and writes exactly 2,000 PNG files named from `dataset/generate.csv`.

The proposal rejects pretrained generative weights, ready-made generation pipelines, copied homework solutions, hidden test data, and existing checkpoints. It allows standard DDPM concepts as implementation references only. It identifies EMA, classifier-free guidance, cosine schedules, DDIM-style sampling, and class-preserving augmentation as possible quality improvements after a valid baseline exists.

## HLD Summary

The HLD defines these modules and boundaries:

- Data and Condition Registry
- Extra Data Intake and Provenance
- Scratch Conditional Denoiser
- Diffusion Process and Sampler
- Training Orchestrator
- Generation Orchestrator
- Output Validator
- 2,000-Image Scoring Adapter
- Submission Packager
- Documentation and Dependency Manifest

Training and generation are offline Python workflows under `scripts/`. Training uses native PyTorch and may be launched through `torchrun` on up to 4 RTX 3090 GPUs. Generation reads `dataset/generate.csv`, loads `model.pth`, and writes `generated_images/`. Validation and scoring are separate from training/generation.

## Design Goals

- Preserve the assignment output contract: exactly 2,000 RGB PNG images, all 64x64, with filenames matching `dataset/generate.csv`.
- Keep the trainable generator path free of pretrained or existing models.
- Make training and generation reproducible through fixed seeds, saved config, saved mappings, logs, and documented commands.
- Support single-GPU execution and explicit multi-GPU execution up to 4 GPUs.
- Provide visible progress and persistent logs for training and generation.
- Keep validation and scoring separate from generation so evaluator-only pretrained models cannot affect produced images.
- Use `unittest` for planned tests unless a later dependency decision adds `pytest`.

## Non-Goals

- Do not implement any code in this design step.
- Do not modify `scoring_program/score.py`.
- Do not make local scoring authoritative over Codabench.
- Do not use pretrained models for training, conditioning, filtering, reranking, or generation.
- Do not require external extra data for the baseline.
- Do not require bitwise-identical output across CPU, single-GPU, and multi-GPU modes.

## Architecture Overview

The implementation is a small Python package-like `scripts/` directory with CLI entry points and shared modules:

| File | Primary Module |
| --- | --- |
| `scripts/brainrot_data.py` | Data and Condition Registry; Extra Data Intake and Provenance helpers |
| `scripts/model.py` | Scratch Conditional Denoiser |
| `scripts/diffusion.py` | Diffusion Process and Sampler |
| `scripts/train.py` | Training Orchestrator |
| `scripts/generate.py` | Generation Orchestrator |
| `scripts/validate_outputs.py` | Output Validator |
| `scripts/score_2000.py` | 2,000-Image Scoring Adapter |
| `scripts/package_submission.py` | Submission Packager |
| `README.md`, `requirements.txt` | Documentation and Dependency Manifest |

Training flow:

1. Parse and validate `dataset/train.csv`.
2. Load 64x64 RGB images from `dataset/trainset/`.
3. Optionally merge approved extra-data records.
4. Convert images to normalized tensors in `[-1, 1]`.
5. Train the scratch denoiser with diffusion noise-prediction loss.
6. Save the final generation checkpoint to `model.pth` without overwriting unless explicitly requested.

Generation flow:

1. Parse and validate `dataset/generate.csv`.
2. Load `model.pth`, including model config and condition mappings.
3. Sample one image per request with deterministic per-request seeds.
4. Write PNG files to `generated_images/` without overwriting unless explicitly requested.
5. Run the output validator before submission or scoring.

## Shared Data Contracts

### Category Contract

Animal order follows the assignment list in `doc/problem-brief.md`:

`shark`, `crocodile`, `frog`, `cat`, `dog`, `capybara`, `elephant`, `bird`, `fish`, `monkey`

Object order follows the assignment list in `doc/problem-brief.md`:

`sneaker`, `airplane`, `coffee cup`, `banana`, `cactus`, `toilet`, `pizza`, `drum`, `car`, `chair`

Derived IDs:

- `animal_id`: index in the animal order.
- `object_id`: index in the object order.
- `pair_id`: `animal_id * 10 + object_id`.
- Null classifier-free-guidance IDs, when enabled, are separate sentinel IDs and are never valid CSV labels.

### CSV Contracts

`dataset/train.csv`:

| Column | Type | Rule |
| --- | --- | --- |
| `id` | string | PNG filename expected under `dataset/trainset/` |
| `animal` | string | Must be one of the 10 known animals |
| `object` | string | Must be one of the 10 known objects |

`dataset/generate.csv`:

| Column | Type | Rule |
| --- | --- | --- |
| `id` | string | Output PNG filename |
| `animal` | string | Must be one of the 10 known animals |
| `object` | string | Must be one of the 10 known objects |
| `prompt` | string | Must equal `a {animal} and a {object}` |

CSV parsing fails on missing required columns, duplicate IDs, unknown labels, empty files, and prompt mismatches. Extra columns are rejected unless a later task explicitly relaxes the schema.

### Runtime Records

`Condition`:

| Field | Type |
| --- | --- |
| `animal` | `str` |
| `object` | `str` |
| `animal_id` | `int` |
| `object_id` | `int` |
| `pair_id` | `int` |

`TrainRecord`:

| Field | Type |
| --- | --- |
| `image_id` | `str` |
| `image_path` | `Path` |
| `condition` | `Condition` |
| `source` | `str`, default `dataset` |

`GenerationRequest`:

| Field | Type |
| --- | --- |
| `image_id` | `str` |
| `prompt` | `str` |
| `condition` | `Condition` |
| `row_index` | `int` |

Image tensor:

- Shape: `[3, 64, 64]` for a single image.
- Batch shape: `[batch, 3, 64, 64]`.
- Type: `torch.float32`.
- Training range: `[-1, 1]`.
- Saved PNG range: `[0, 255]`, RGB mode.

### Checkpoint Contract

`model.pth` is a PyTorch checkpoint dictionary:

| Key | Meaning |
| --- | --- |
| `format_version` | Integer checkpoint format version, initially `1` |
| `model_state_dict` | Scratch denoiser weights used for generation |
| `ema_state_dict` | EMA weights when EMA is enabled, otherwise absent or `None` |
| `model_config` | Architecture configuration needed to rebuild the model |
| `diffusion_config` | Schedule and training timestep configuration |
| `animals` | Ordered animal labels |
| `objects` | Ordered object labels |
| `seed` | Training seed |
| `train_step` | Final or selected training step |
| `uses_pretrained_generator_weights` | Must be `False` |

Generation uses `ema_state_dict` when present and requested, otherwise `model_state_dict`.

### Runtime Configuration Contract

All entry points support explicit paths and fail before writing if an output target already exists, unless `--overwrite` is passed.

Device configuration:

- `--devices` accepts comma-separated CUDA device indices, such as `1,2,3,4`.
- More than 4 devices is an error.
- Unavailable device IDs are an error.
- No training or generation code may hardcode `cuda:0`.
- CPU mode may be supported only for smoke checks and must be clearly labeled.

### Progress Log Contract

Training and generation write JSONL or plain text logs containing:

- Timestamp.
- Seed.
- Device list.
- Command/config.
- Elapsed time.
- Estimated remaining time when available.
- Current step or generated count.
- Loss or save event for training.
- Output path for generation.

Exact log format is an implementation choice, but every field above must be present.

## Module Designs

### Data and Condition Registry

#### Responsibility

Parse assignment CSVs, validate category and prompt contracts, build stable animal/object/pair mappings, load training images, and expose records usable by training and generation.

#### Non-Responsibility

It does not train models, sample images, score generated images, write final outputs, package submissions, or admit external data without the extra-data intake checks.

#### Inputs and Outputs

Inputs:

- `dataset/train.csv`
- `dataset/generate.csv`
- `dataset/trainset/*.png`
- Approved extra-data records from Extra Data Intake and Provenance

Outputs:

- `TrainRecord` list or dataset.
- `GenerationRequest` list.
- Shared ordered category mappings.
- PyTorch dataset returning `(image_tensor, condition)`.

#### Public Interface

CLI exposure is through `train.py`, `generate.py`, and `validate_outputs.py`.

Planned Python interface:

- `load_train_records(train_csv: Path, image_dir: Path) -> list[TrainRecord]`
- `load_generation_requests(generate_csv: Path) -> list[GenerationRequest]`
- `build_condition(animal: str, object_name: str) -> Condition`
- `BrainrotDataset(records: Sequence[TrainRecord], augment: bool, image_size: int = 64)`
- `validate_prompt(animal: str, object_name: str, prompt: str) -> None`

#### Data Structures

Use `dataclasses.dataclass(frozen=True)` for `Condition`, `TrainRecord`, and `GenerationRequest`. Use assignment-order tuples for label vocabularies. Use `dict[str, int]` maps derived from those tuples.

#### Internal Design

CSV parsing uses Python `csv.DictReader`. Path handling uses `pathlib.Path`. Image inspection/loading uses PIL. Dataset batching uses PyTorch `Dataset`.

Training transforms:

- Open image.
- Convert to RGB.
- Verify size `(64, 64)`.
- Convert to tensor.
- Normalize to `[-1, 1]`.
- Apply class-preserving augmentation only when enabled.

Allowed baseline augmentations are horizontal flip and mild color jitter only if implemented without changing image size or labels. Any augmentation default must be documented in `README.md`.

#### Algorithm Details

CSV validation:

```text
read header
require exact schema
for each row:
    reject duplicate id
    reject unknown animal/object
    build condition from assignment-order mappings
    if generate row:
        require prompt == "a {animal} and a {object}"
    if train row:
        require image path exists
        inspect image as PNG/RGB/64x64
return records in file order
```

#### Dependencies

Standard library `csv`, `dataclasses`, `pathlib`; PIL; PyTorch dataset utilities.

#### Failure Handling

Raise row-specific `ValueError` for schema, duplicate ID, label, or prompt failures. Raise `FileNotFoundError` for missing images. Raise image validation errors for unreadable files, non-PNG files, wrong dimensions, or non-RGB images that cannot be converted safely.

#### Independent Test Plan

- Valid train and generate fixture parsing.
- Duplicate ID rejection.
- Unknown animal/object rejection.
- Missing image rejection.
- Wrong image dimensions rejection.
- Prompt mismatch rejection, including `coffee cup`.
- Deterministic mapping independent of CSV row order.

#### Open Questions

None blocking. Extra external data remains disabled unless provided through the extra-data module.

### Extra Data Intake and Provenance

#### Responsibility

Admit optional external training records only when assignment-allowed, reproducible, manually/provenance-labeled, and mapped into the same 10x10 condition space.

#### Non-Responsibility

It does not download data by default, infer labels, use pretrained models, generate synthetic training images, or decide final model quality.

#### Inputs and Outputs

Inputs:

- Optional manifest path passed by `--extra_manifest`.
- Extra image files referenced by the manifest.

Outputs:

- Validated `TrainRecord` entries with `source != "dataset"`.
- A provenance report suitable for `README.md`.

#### Public Interface

- `load_extra_records(manifest_csv: Path) -> list[TrainRecord]`
- `validate_extra_manifest(manifest_csv: Path) -> list[str]`

Training CLI flag:

- `--extra_manifest PATH`, default absent.

#### Data Structures

Minimum extra-data manifest columns:

| Column | Rule |
| --- | --- |
| `image_path` | Existing local file path |
| `animal` | Known animal |
| `object` | Known object |
| `source` | Source URL/path/name |
| `retrieval_date` | Date or Unknown if source is local historical data |
| `license_or_allowance` | Evidence that use is allowed |
| `label_author_or_method` | Must not be a pretrained model |
| `image_hash` | Hash recorded for reproducibility |
| `preprocessing_notes` | Resize/crop notes or `none` |

#### Internal Design

The module validates the manifest before any extra record enters training. It rejects pretrained-model-generated images and pretrained-model-generated labels based on explicit manifest fields or provenance text. If the manifest is absent, the module returns no records.

#### Algorithm Details

```text
if no manifest:
    return []
read manifest
require minimum columns
for each row:
    validate source/provenance fields are non-empty
    reject pretrained-generated image/label evidence
    validate condition labels
    validate image exists, PNG/RGB/64x64 or documented preprocessing output
    validate hash if implementation supports hash checking
return records
```

#### Dependencies

Data and Condition Registry, standard library CSV/path/hashlib, PIL.

#### Failure Handling

Fail closed. Any missing provenance, unknown label, invalid image, ambiguous label method, or non-reproducible path rejects the extra record. A malformed manifest fails the whole extra-data intake before training starts.

#### Independent Test Plan

- Default no-manifest path returns no extra records.
- Valid manifest fixture is accepted.
- Missing provenance is rejected.
- Unknown label is rejected.
- Pretrained-model-generated image or label is rejected.
- Wrong image dimensions are rejected.

#### Open Questions

The exact external extra-data source is not defined. Baseline training uses repository data plus class-preserving augmentation.

### Scratch Conditional Denoiser

#### Responsibility

Predict diffusion noise for noisy 64x64 RGB images conditioned on timestep, animal ID, object ID, and pair ID.

#### Non-Responsibility

It does not own the diffusion schedule, training loop, sampling loop, data parsing, checkpoint policy, pretrained weights, or evaluator models.

#### Inputs and Outputs

Inputs:

- `x_t`: `[batch, 3, 64, 64]` noisy image tensor.
- `timesteps`: `[batch]` integer timestep tensor.
- `conditions`: animal/object/pair IDs or null CFG IDs.

Output:

- Predicted noise tensor `[batch, 3, 64, 64]`.

#### Public Interface

- `UNetConfig`
- `ConditionalUNet(config: UNetConfig)`
- `ConditionalUNet.forward(x_t, timesteps, animal_ids, object_ids, pair_ids) -> torch.Tensor`

No constructor argument may load pretrained weights or checkpoints.

#### Data Structures

`UNetConfig` includes image channels, base channel count, channel multipliers, number of residual blocks, embedding dimension, dropout, number of animals, objects, and pairs, plus CFG null-token support.

Exact numeric defaults are tuning parameters and must be recorded in the checkpoint and `README.md` when implementation begins.

#### Internal Design

Use a scratch U-Net:

- Sinusoidal timestep embedding followed by learned MLP.
- Learned animal, object, and pair embeddings.
- Combined conditioning embedding injected into residual blocks.
- Downsample path for 64 -> 32 -> 16 -> 8 resolution.
- Bottleneck residual/attention block only if implemented from scratch.
- Upsample path with skip connections.
- Final convolution to 3 channels.

GroupNorm and SiLU are acceptable PyTorch primitives. Attention is optional and must be self-implemented with PyTorch tensor ops if added.

#### Algorithm Details

The forward path combines timestep and condition embeddings into a single conditioning vector. CFG training is supported by replacing condition IDs with null IDs according to training-time condition-drop probability. CFG sampling is handled by the sampler, not by this module.

#### Dependencies

PyTorch `torch` and `torch.nn`.

#### Failure Handling

Reject invalid tensor shapes, invalid label IDs, and unsupported image sizes with clear errors during smoke checks. Avoid downloads or filesystem reads in the model constructor.

#### Independent Test Plan

- Constructor creates random-initialized parameters without reading files.
- Forward output shape equals input shape.
- Timestep and condition changes affect output for fixed `x_t`.
- Backward pass produces finite gradients.
- Constructor does not call OpenCLIP, TorchVision pretrained weights, or checkpoint loading.

#### Open Questions

Exact architecture width/depth defaults are tuning choices. They must fit the under-8-hour 4-GPU target after implementation smoke tests.

### Diffusion Process and Sampler

#### Responsibility

Own the diffusion schedule, `q_sample`, noise-prediction loss, EMA-compatible sampling contract, CFG mixing, and DDPM/DDIM reverse sampling.

#### Non-Responsibility

It does not parse data, write PNGs, own model weights, decide devices, or log training progress.

#### Inputs and Outputs

Training inputs:

- Clean image batch `x_0`.
- Conditions.
- Scratch denoiser.

Training output:

- Scalar MSE noise-prediction loss.

Sampling inputs:

- Scratch denoiser.
- Generation conditions.
- Seeded random noise.
- Sampling configuration.

Sampling output:

- Generated tensor batch in image space.

#### Public Interface

- `DiffusionConfig`
- `GaussianDiffusion(config: DiffusionConfig)`
- `GaussianDiffusion.training_loss(model, x_0, conditions, generator=None) -> torch.Tensor`
- `GaussianDiffusion.sample(model, conditions, shape, sampler, steps, guidance_scale, generator=None) -> torch.Tensor`
- `build_schedule(config: DiffusionConfig) -> ScheduleTensors`

#### Data Structures

`DiffusionConfig` includes:

- `image_size`
- `train_timesteps`
- `schedule`: `linear` or `cosine`
- `prediction_target`: initially `epsilon`
- `sampler`: `ddpm` or `ddim`
- `sampling_steps`
- `cfg_dropout`
- `guidance_scale`

#### Internal Design

Precompute schedule tensors on initialization and move them to the active device with the batch. Use MSE between sampled noise and predicted noise. Use deterministic sampling generators for smoke tests and per-request seeds.

Initial implementation should support DDPM sampling first. DDIM may be added behind the same sampler interface once the DDPM path passes smoke tests.

#### Algorithm Details

Training loss:

```text
t ~ Uniform({0, ..., T-1})
epsilon ~ Normal(0, I)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
epsilon_hat = model(x_t, t, maybe_dropped_condition)
loss = mean((epsilon - epsilon_hat)^2)
```

CFG sampling:

```text
epsilon_uncond = model(x_t, t, null_condition)
epsilon_cond = model(x_t, t, condition)
epsilon = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```

Complexity:

- Training step: one model forward and backward per batch.
- DDPM sampling: `sampling_steps` model forwards per generated batch, doubled when CFG is active.

#### Dependencies

PyTorch tensor operations and Scratch Conditional Denoiser.

#### Failure Handling

Reject invalid schedule values, non-monotonic cumulative alphas, invalid sampling step counts, NaN/Inf losses, and unsupported sampler names.

#### Independent Test Plan

- Schedule tensors are finite and valid.
- Cumulative alphas are monotonic.
- `q_sample` matches direct formula on tiny tensors.
- Loss is scalar and finite.
- Seeded sampling is repeatable on the same device mode.
- Pixel conversion before save clamps to valid image range.

#### Open Questions

The final default schedule and sampling-step count are tuning choices. The design supports both linear/cosine schedules and DDPM/DDIM samplers so they can be compared without changing module boundaries.

### Training Orchestrator

#### Responsibility

Run the training lifecycle: parse config, validate devices, set seeds, load data, initialize model/diffusion/optimizer, run training, maintain EMA when enabled, save checkpoints and `model.pth`, and report progress.

#### Non-Responsibility

It does not generate final submission images, run local scoring, package submissions, or modify scorer files.

#### Inputs and Outputs

Inputs:

- Training CSV and image directory.
- Optional extra-data manifest.
- Runtime, model, diffusion, optimizer, and device settings.

Outputs:

- `model.pth`.
- Optional checkpoints under `checkpoints/`.
- Logs and sample grids under `runs/`.

#### Public Interface

Primary command:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --output_model model.pth
```

Multi-GPU launch shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --output_model model.pth
```

Planned flags:

- `--train_csv`
- `--image_dir`
- `--extra_manifest`
- `--output_model`
- `--run_dir`
- `--seed`
- `--devices`
- `--max_steps`
- `--batch_size`
- `--learning_rate`
- `--weight_decay`
- `--amp`
- `--grad_accum_steps`
- `--ema`
- `--ema_decay`
- `--schedule`
- `--train_timesteps`
- `--cfg_dropout`
- `--save_every_steps`
- `--sample_every_steps`
- `--overwrite`
- `--cpu_smoke`

#### Data Structures

Training config is a serializable dictionary saved to logs and `model.pth`. Distributed state is derived from `LOCAL_RANK`, `RANK`, and `WORLD_SIZE` when launched by `torchrun`.

#### Internal Design

Use native PyTorch DistributedDataParallel for multi-GPU training. Single-process training remains the default when no distributed environment is detected. Validate that the effective GPU count is at most 4.

Training loop:

1. Set seeds.
2. Validate data and optional extra-data manifest.
3. Build dataset and sampler.
4. Build model and diffusion from config.
5. Wrap model in DDP when distributed.
6. Run optimization loop until `--max_steps` optimizer updates have completed, with progress reporting.
7. Update EMA after optimizer steps when enabled.
8. Save checkpoints periodically if configured.
9. Save final `model.pth` atomically.

#### Algorithm Details

`model.pth` should be written through a temporary file in the same directory and renamed after successful serialization. Existing `model.pth`, checkpoint files, or run logs are not overwritten unless `--overwrite` is passed.

#### Dependencies

Data and Condition Registry, Extra Data Intake and Provenance, Scratch Conditional Denoiser, Diffusion Process and Sampler, PyTorch distributed APIs, `tqdm` for progress.

#### Failure Handling

Fail before training if data validation fails, output files already exist without `--overwrite`, more than 4 GPUs are requested, CUDA devices are unavailable, pretrained checkpoint arguments are supplied, or distributed initialization is inconsistent.

#### Independent Test Plan

- One-step training smoke on a tiny subset.
- Finite loss and gradient presence.
- Progress log contains required fields.
- Single-GPU fallback path can be initialized.
- Distributed launch config rejects more than 4 GPUs.
- Existing output path fails without `--overwrite`.
- No hardcoded `cuda:0`.
- No pretrained/existing model load in training.

#### Open Questions

Exact hyperparameter defaults remain to be finalized during implementation smoke tests and recorded in `README.md`.

### Generation Orchestrator

#### Responsibility

Load `model.pth`, read `dataset/generate.csv`, sample one image per request, and write final PNGs to `generated_images/`.

#### Non-Responsibility

It does not train, score, filter, rerank, or package images. It does not use pretrained models.

#### Inputs and Outputs

Inputs:

- `model.pth`
- `dataset/generate.csv`
- Sampling config and device config

Outputs:

- `generated_images/*.png`, exactly one file per generation request.
- Generation log under `runs/` unless configured otherwise.

#### Public Interface

Primary command:

```bash
python scripts/generate.py --model model.pth --generate_csv dataset/generate.csv --output_dir generated_images
```

Multi-GPU launch shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/generate.py --model model.pth --generate_csv dataset/generate.csv --output_dir generated_images
```

Planned flags:

- `--model`
- `--generate_csv`
- `--output_dir`
- `--run_dir`
- `--seed`
- `--devices`
- `--sampler`
- `--sampling_steps`
- `--guidance_scale`
- `--batch_size`
- `--use_ema`
- `--overwrite`
- `--cpu_smoke`
- `--limit`, for temporary smoke generation only

#### Data Structures

Generation uses `GenerationRequest` records in CSV order. Per-request seed is `base_seed + row_index`, so multi-GPU partitioning does not depend on completion order.

#### Internal Design

Load checkpoint config and verify category mapping matches the current assignment mapping. Partition requests by distributed rank when launched under `torchrun`; otherwise process all requests in one process. Each rank writes a disjoint filename set.

Output directory handling:

- If final `generated_images/` exists and is non-empty, fail unless `--overwrite` is passed.
- For smoke runs, require a temporary output directory or `--limit`.
- Write RGB PNGs only.

#### Algorithm Details

```text
requests = load_generation_requests(generate_csv)
rank_requests = requests[rank::world_size]
for batch in rank_requests:
    seeds = base_seed + row_index
    sample tensors with diffusion sampler
    convert [-1, 1] tensors to RGB uint8
    save as output_dir / image_id
barrier if distributed
rank 0 reports total expected count
```

#### Dependencies

Data and Condition Registry, Scratch Conditional Denoiser, Diffusion Process and Sampler, PIL or TorchVision image saving, PyTorch runtime, `tqdm`.

#### Failure Handling

Fail on missing checkpoint, incompatible checkpoint schema, invalid generation CSV, output collisions, more than 4 GPUs, missing CUDA for requested CUDA mode, invalid sampling config, or image save errors.

#### Independent Test Plan

- Temporary generation of a few requests.
- Requested filenames are preserved.
- Same seed and same device mode repeat outputs.
- Existing final output dir fails without `--overwrite`.
- Multi-GPU partition simulation has no missing or duplicate filenames.
- No pretrained/existing model usage.

#### Open Questions

Final guidance scale and sampling-step defaults are quality tuning choices.

### Output Validator

#### Responsibility

Check that generated files satisfy the assignment output contract.

#### Non-Responsibility

It does not generate images, alter images, score images, or package submissions.

#### Inputs and Outputs

Inputs:

- `dataset/generate.csv`
- Generated image directory

Outputs:

- Pass/fail exit status.
- Actionable validation report.

#### Public Interface

```bash
python scripts/validate_outputs.py --generate_csv dataset/generate.csv --image_dir generated_images --expected_count 2000
```

Planned flags:

- `--generate_csv`
- `--image_dir`
- `--expected_count`
- `--strict_prompt`

#### Data Structures

Expected filename set comes directly from `dataset/generate.csv`. Actual filename set comes from `*.png` files directly under `image_dir`.

#### Internal Design

Validate set equality before image inspection so missing and extra files are reported clearly. Then inspect each expected file with PIL.

#### Algorithm Details

```text
expected = ids from generate_csv
actual = png filenames in image_dir
require len(expected) == expected_count
require actual == expected
for each expected filename:
    open image
    require PNG format
    require size == (64, 64)
    require mode == RGB
```

#### Dependencies

Standard library CSV/path handling and PIL.

#### Failure Handling

Return nonzero on any violation and include missing filenames, extra filenames, invalid image paths, wrong dimensions, wrong modes, and invalid formats.

#### Independent Test Plan

- Passing controlled fixture.
- Missing file fixture.
- Extra file fixture.
- Wrong extension fixture.
- Invalid PNG fixture.
- Wrong dimension fixture.
- Wrong color mode fixture.
- Duplicate/missing generated filename case.

#### Open Questions

None.

### 2,000-Image Scoring Adapter

#### Responsibility

Compute local development FID and CLIP-T for exactly the 2,000 generated submission images, without raw validation images and without influencing generation.

#### Non-Responsibility

It does not train, condition, filter, rerank, or regenerate images. It is not the official grading source.

#### Inputs and Outputs

Inputs:

- `generated_images/`
- `dataset/generate.csv`
- `scoring_program/input/ref/test_mu.npy`
- `scoring_program/input/ref/test_sigma.npy`
- Evaluator model weights for Inception/OpenCLIP, used only for scoring

Outputs:

- Score JSON report, default under a run-specific scoring directory.

#### Public Interface

```bash
python scripts/score_2000.py --image_dir generated_images --generate_csv dataset/generate.csv --ref_mu scoring_program/input/ref/test_mu.npy --ref_sigma scoring_program/input/ref/test_sigma.npy --scores fid clip_t
```

Planned flags:

- `--image_dir`
- `--generate_csv`
- `--ref_mu`
- `--ref_sigma`
- `--scores`
- `--output`
- `--device`
- `--batch_size`
- `--num_workers`
- `--overwrite`

#### Data Structures

Score report:

| Field | Meaning |
| --- | --- |
| `FID` | Present if FID was requested and completed |
| `CLIP_T` | Present if CLIP-T was requested and completed |
| `num_images` | Must be 2,000 for final generated set |
| `image_dir` | Scored image directory |
| `generate_csv` | Prompt source |
| `evaluator_models` | Inception/OpenCLIP model identifiers |
| `notes` | Fixture limitations and command metadata |

#### Internal Design

Reuse the metric semantics in `scoring_program/score.py`, but avoid its raw-reference-image requirement for CLIP-I and its 3,000-image default. FID uses generated image features against `test_mu.npy` and `test_sigma.npy`. CLIP-T uses generated images and prompts from `generate.csv`.

This module may load pretrained evaluator weights. Training and generation modules must not import this module.

#### Algorithm Details

FID:

```text
validate output contract for 2,000 images
extract Inception features for generated images
compute generated mean/covariance
FID = ||mu_ref - mu_gen||^2 + trace(sigma_ref + sigma_gen - 2 * sqrtm(sigma_ref * sigma_gen))
```

CLIP-T:

```text
validate output contract for 2,000 images
for each generation row:
    load generated image
    tokenize prompt
    compute normalized image/text embeddings
    score = cosine similarity
CLIP_T = mean(score)
```

#### Dependencies

Output Validator, NumPy, SciPy, PyTorch, TorchVision, OpenCLIP, PIL, tqdm.

#### Failure Handling

Fail if generated images do not pass validation, reference statistics are missing for FID, prompts are missing for CLIP-T, evaluator weights are unavailable, CUDA is requested but unavailable, or output score file exists without `--overwrite`.

#### Independent Test Plan

- Refuse missing reference statistics for FID.
- Refuse missing prompts for CLIP-T.
- Confirm exactly 2,000 target images for final scoring.
- Confirm raw validation images are not required for CLIP-T.
- Confirm evaluator-only model usage is isolated from training/generation.
- Confirm unrun scores are never reported as verified.

#### Open Questions

Network availability for first-time evaluator weight download is unknown. Cached weights or approved network access may be needed for local scoring.

### Submission Packager

#### Responsibility

Assemble the E3 archive once final artifacts already exist and validation has passed.

#### Non-Responsibility

It does not train, generate, fix invalid outputs, or upload to Codabench/E3.

#### Inputs and Outputs

Inputs:

- `generated_images/`
- `scripts/`
- `model.pth`
- `README.md`
- `requirements.txt`
- Student ID `314511048`

Output:

- `HW6_314511048.zip`

#### Public Interface

```bash
python scripts/package_submission.py --student_id 314511048 --output HW6_314511048.zip
```

Planned flags:

- `--student_id`
- `--generated_dir`
- `--scripts_dir`
- `--model`
- `--readme`
- `--requirements`
- `--output`
- `--overwrite`

#### Data Structures

Archive top-level entries:

- `generated_images/`
- `scripts/`
- `model.pth`
- `README.md`
- `requirements.txt`

#### Internal Design

Use Python standard library `zipfile` for reproducible packaging. Run or invoke Output Validator before packaging. Exclude development-only directories such as `checkpoints/`, `runs/`, and local score reports unless explicitly requested by a later task.

#### Algorithm Details

```text
validate required paths exist
validate generated_images contract
reject existing output unless overwrite
write zip with required top-level entries
report archive path and file count
```

#### Dependencies

Output Validator and standard library `zipfile`, `pathlib`.

#### Failure Handling

Fail on missing required files, invalid generated images, wrong archive name, existing archive without `--overwrite`, or unreadable input files.

#### Independent Test Plan

- Archive name is exactly `HW6_314511048.zip`.
- Required paths are present.
- Missing `model.pth` fails.
- Missing `requirements.txt` fails.
- Archive contains expected top-level entries.
- Existing archive fails without `--overwrite`.

#### Open Questions

Whether large model weights require cloud-link handling is unknown until the final `model.pth` size exists.

### Documentation and Dependency Manifest

#### Responsibility

Record setup, dependencies, training command, generation command, validation command, scoring command, seeds, hardware assumptions, and reproduction notes.

#### Non-Responsibility

It does not replace runnable tests, generate outputs, or claim unrun commands were run.

#### Inputs and Outputs

Inputs:

- Final implementation commands.
- Final dependency choices.
- Final training/generation settings and seeds.
- Extra-data provenance if external data is used.

Outputs:

- Updated `README.md`.
- `requirements.txt`.

#### Public Interface

Files:

- `README.md`
- `requirements.txt`

Recommended post-implementation commands to document only after they exist:

```bash
python -m pip install -r requirements.txt
python -m unittest discover
python -m compileall scoring_program scripts
```

#### Data Structures

`requirements.txt` should list direct runtime dependencies for training, generation, validation, scoring, and packaging. Versions are Unknown until implementation stabilizes.

#### Internal Design

Keep README commands non-destructive by default. Any command that overwrites `model.pth`, `generated_images/`, checkpoints, run logs, scores, or archives must require an explicit overwrite flag.

#### Algorithm Details

No algorithm. Documentation must distinguish verified commands from planned commands.

#### Dependencies

None beyond the implemented project files.

#### Failure Handling

Manual/static review fails if README claims unrun metrics, omits reproduction seeds, omits generated-output validation, documents `--score` instead of the script-compatible `--scores` where referencing `score.py`, or omits no-pretrained constraints.

#### Independent Test Plan

- README includes setup, train, generate, validate, scoring, seed, hardware, and extra-data notes.
- `requirements.txt` exists before install commands are treated as runnable.
- No prohibited high-level generation pipeline dependencies are introduced.
- Commands are labeled planned/run accurately.

#### Open Questions

Exact dependency versions remain unknown until implementation and environment checks.

## Cross-Module Contracts

- Data and Condition Registry is the only source of animal/object/pair mappings.
- Training saves the mappings and configs needed by generation inside `model.pth`.
- Generation rejects checkpoints with incompatible or missing mappings.
- Scoring Adapter may use pretrained evaluator models, but training and generation must not import or depend on it.
- Output Validator is the gate before scoring and packaging.
- All write-capable CLIs reject existing outputs unless `--overwrite` is passed.
- Distributed training and generation must enforce a maximum of 4 GPUs.
- Progress logging must not be required at import time; logs are created only when a CLI runs.
- Hidden test data must never enter training, validation, generation, or packaging.

## End-to-End Workflow

1. Install dependencies after `requirements.txt` exists:

```bash
python -m pip install -r requirements.txt
```

2. Run unit/smoke checks after tests exist:

```bash
python -m unittest discover
python -m compileall scoring_program scripts
```

3. Train from scratch:

```bash
python scripts/train.py --train_csv dataset/train.csv --image_dir dataset/trainset --output_model model.pth
```

4. Generate final images:

```bash
python scripts/generate.py --model model.pth --generate_csv dataset/generate.csv --output_dir generated_images
```

5. Validate output contract:

```bash
python scripts/validate_outputs.py --generate_csv dataset/generate.csv --image_dir generated_images --expected_count 2000
```

6. Optionally score locally when evaluator assets and weights are available:

```bash
python scripts/score_2000.py --image_dir generated_images --generate_csv dataset/generate.csv --ref_mu scoring_program/input/ref/test_mu.npy --ref_sigma scoring_program/input/ref/test_sigma.npy --scores fid clip_t
```

7. Package E3 submission:

```bash
python scripts/package_submission.py --student_id 314511048 --output HW6_314511048.zip
```

These commands are planned design commands until implementation exists and they are run.

## Test Strategy Mapping

| Test-plan requirement | Coverage in this design |
| --- | --- |
| Import smoke without side effects | Every module separates importable helpers from CLI execution. |
| Dataset smoke | Data and Condition Registry independent tests. |
| Diffusion/model smoke | Scratch Conditional Denoiser and Diffusion Process independent tests. |
| Generation smoke | Generation Orchestrator temporary-output tests. |
| Output validator smoke | Output Validator pass/fail fixture tests. |
| CSV schemas and mappings | Shared CSV contracts and Data and Condition Registry tests. |
| Extra-data validation | Extra Data Intake and Provenance tests; default disabled. |
| Random initialization/no pretrained | Scratch Conditional Denoiser tests and cross-module no-pretrained contract. |
| Diffusion schedule/noise/loss/sampling | Diffusion Process tests and formula oracle. |
| Training seed/progress/checkpoint/multi-GPU | Training Orchestrator tests. |
| Generation ordering/partitioning/progress | Generation Orchestrator tests. |
| 2,000-image scoring adapter | Scoring Adapter tests and failure handling. |
| Submission packaging | Submission Packager tests. |
| README and requirements | Documentation and Dependency Manifest tests. |
| Integration: dataset to training batch | Data registry + model/diffusion one-step training test. |
| Integration: save/load to generation | Training Orchestrator + Generation Orchestrator smoke. |
| Integration: generation to validator | Generation smoke followed by Output Validator. |
| Integration: multi-GPU request partition | Generation Orchestrator partition tests; Training Orchestrator DDP launch checks. |
| Integration: scoring fixture checks | Scoring Adapter missing-fixture tests. |
| Golden train/generate rows | Data and Condition Registry golden tests. |
| Prompt consistency | Data and Condition Registry prompt validator. |
| Wrong dimensions/modes/extensions | Output Validator and Data Registry image tests. |
| Diffusion tiny tensor oracle | Diffusion Process independent tests. |
| Archive naming | Submission Packager tests. |
| Property: mapping row-order independence | Data and Condition Registry property-style tests with deterministic shuffles. |
| Property: category validation | Data and Condition Registry randomized valid/invalid labels. |
| Property: output filename set | Output Validator randomized remove/add/duplicate fixtures. |
| Property: schedule validity | Diffusion Process randomized timestep checks. |
| Property: seed determinism | Diffusion and Generation smoke tests on same device mode. |
| Property: multi-GPU partition | Generation partition tests. |
| Cross-device bitwise determinism | Not required; documented as non-goal. |
| Edge cases | Covered in each module's failure handling and independent tests. |
| Performance benchmarks | Training and Generation progress logs expose throughput, elapsed time, ETA, and GPU count. |
| Evaluator command caveats | Scoring Adapter avoids `--score`/`--scores` mismatch by using its own CLI. |
| Regression: 3,000 vs 2,000 | Validator and Scoring Adapter default to 2,000 final images. |
| Regression: hardcoded `cuda:0` | Runtime configuration contract and orchestrator tests. |
| Regression: overwrite protection | All write-capable CLIs require `--overwrite`. |
| Regression: no-pretrained path | Model, training, generation, and extra-data tests. |
| Manual verification | Documentation module and README review requirements. |
| Minimum done criteria | Quality Gates section below. |

## Quality Gates

Design-time status:

- `doc/proposal.md`, `doc/high-level-design.md`, and `doc/test-plan.md` exist and were used.
- No implementation tests were run because no implementation exists yet.
- No scorer command was run because generated images and complete local scorer fixtures are absent.

Implementation-ready gates:

- `python -m unittest discover` passes after tests are added.
- `python -m compileall scoring_program scripts` passes after scripts are added.
- Dataset parser tests pass for valid rows, invalid rows, duplicate IDs, unknown labels, prompt consistency, and image existence/dimensions.
- Model/diffusion smoke passes with finite loss and gradients from random initialization.
- No-pretrained-path checks pass for training and generation.
- Training smoke writes progress/logs without touching final `generated_images/`.
- Generation smoke writes a temporary valid PNG set.
- Output Validator passes the final `generated_images/` and fails controlled bad fixtures.
- Final `generated_images/` contains exactly 2,000 RGB PNG files, all 64x64, with filenames equal to `dataset/generate.csv`.
- Multi-GPU behavior is tested with `torchrun` or blocked by documented hardware availability.
- Local 2,000-image FID/CLIP-T scoring is run or explicitly blocked with the missing reason.
- `README.md`, `requirements.txt`, `scripts/`, `model.pth`, and `generated_images/` exist before packaging.
- `HW6_314511048.zip` contains required top-level entries.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Small dataset causes overfitting or low diversity | Use output validation first, then tune augmentation, model size, EMA, CFG, schedule, and sampler. |
| CFG improves CLIP-T but hurts FID | Keep guidance scale configurable and score multiple generated sets. |
| Multi-GPU generation drops or duplicates files | Partition by row index and validate filename set after generation. |
| Hardcoded device assumptions fail on busy GPU 0 | Require explicit device handling and reject hardcoded `cuda:0` in tests/review. |
| Local scorer fixture mismatch | Use dedicated 2,000-image adapter and keep Codabench authoritative. |
| Extra data violates reproducibility or restrictions | Default external extra data off; require manifest/provenance and reject pretrained-generated/labeled records. |
| Unverified commands become misleading documentation | README must label planned, run, and blocked commands accurately. |
| Existing generated artifacts are overwritten | All writer CLIs fail on existing outputs unless `--overwrite` is explicit. |

## Assumptions

- PyTorch is the implementation base.
- The first complete baseline uses only `dataset/` plus optional class-preserving augmentation.
- Assignment-order category lists are the stable source for condition IDs.
- `model.pth` stores enough config and mapping state for generation.
- `unittest` is the default test framework because no dependency manifest exists yet.
- Exact quality-tuned hyperparameters are not known until implementation smoke tests and development scoring.
- Evaluator-only pretrained Inception/OpenCLIP weights are allowed for local scoring reports, not generation.
- Codabench remains authoritative for final FID and CLIP-T.

## Open Questions

- Exact architecture width/depth defaults.
- Exact training batch size, step count, learning rate, EMA decay, schedule, guidance scale, and sampling-step defaults.
- Exact external extra-data source, if any.
- Exact dependency versions for `requirements.txt`.
- Whether first-time evaluator weight downloads will be allowed or cached for local scoring.
- Whether final `model.pth` size requires cloud-link handling in the E3 submission.
