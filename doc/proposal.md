# Proposal: Scratch Conditional Brainrot Image Generator

## Objective

Build a reproducible, from-scratch solution for HW6 Brainrot Image Generation that trains a conditional image generation model and produces the required 2,000 64x64 PNG images for `dataset/generate.csv`.

The implementation must not copy public homework solutions, import ready-made generation pipelines, use pretrained generative weights, or depend on existing model checkpoints. Public diffusion implementations may be used only as conceptual references for standard DDPM components.

## Source Inputs

- User objective: propose a solution based on prior search inspiration, with no copy-paste and no existing models.
- `doc/problem-brief.md`: assignment requirements, dataset details, constraints, grading thresholds, and deliverables.
- `doc/repo-map.md`: repository state, available dataset/scorer files, missing implementation files.
- `doc/quality-gates.md`: discovered validation gaps and scorer command issues.
- Prior search references used only for implementation inspiration:
  - `tcapelle/Diffusion-Models-pytorch`: compact conditional DDPM shape with CFG and EMA.
  - Hugging Face Annotated Diffusion: conceptual DDPM training/sampling explanation.
  - `openai/improved-diffusion`: 64x64 hyperparameter ideas such as cosine schedule, EMA sampling, class conditioning, and microbatching.
  - `TeaPearce/Conditional_Diffusion_MNIST`: minimal classifier-free guidance concept.

## Current Project State

- Dataset files exist locally:
  - `dataset/train.csv` with 4,799 training records.
  - `dataset/generate.csv` with 2,000 generation requests.
  - `dataset/trainset/` with 4,799 64x64 PNG training images.
- Local scorer files partially exist under `scoring_program/`.
- No model, training script, generation script, dependency manifest, README instructions, generated images, or model weights exist yet.
- No tests or smoke checks exist yet.
- `scoring_program/score.py` supports `--scores`, while `scoring_manual.txt` documents `--score`.
- Local scorer reference prompt/image files are incomplete for full CLIP scoring.

## Problem Summary

The assignment requires generating Brainrot images for 100 animal-object combinations. The model must learn from a small 4,799-image dataset and generate 20 images per condition for the 2,000 requested IDs in `generate.csv`.

Official grading uses FID and CLIP-T, each worth 50%. The output contract is strict: exactly 2,000 PNG images, correct filenames, 64x64 resolution, and condition alignment with `generate.csv`.

## Constraints

- Main generator must be trained from scratch.
- No pretrained UNet, Transformer, Diffusion Model, generative model weights, or ready-made high-level generation pipelines.
- Pretrained CLIP or VAE may only be auxiliary if used; this proposal avoids using them in the generator path.
- Training and generation must be reproducible by the teaching assistants.
- Submission must include `generated_images/`, `scripts/`, `model.pth`, `README.md`, and `requirements.txt`.
- Official image generation target is 2,000 images, even though the local scorer config currently references 3,000.

## Proposed Approach

Implement a scratch conditional DDPM in PyTorch for 64x64 RGB images.

Use a custom U-Net denoiser trained to predict diffusion noise from:

- Noisy image tensor.
- Diffusion timestep embedding.
- Animal/object condition embedding.

Use the Brainrot training set only for the first complete version. Extra data is allowed by the assignment, but it adds provenance and reproducibility burden and should be considered only after a working baseline is validated.

The first target should be a clean, reproducible baseline that can produce valid `generated_images/`. Then improve the same model family with assignment-safe upgrades: stronger conditioning, classifier-free guidance, EMA weights, cosine noise schedule, and DDIM-style faster sampling.

## Algorithm Strategy

Baseline method:

- Train a pixel-space conditional DDPM on 64x64 RGB images.
- Represent each animal-object pair as one of 100 class IDs.
- Use a small-to-medium U-Net sized for 12GB VRAM constraints suggested by the assignment baselines.
- Train with MSE noise-prediction loss.
- Sample with the trained EMA or latest model and write images using exact IDs from `generate.csv`.

Intended optimized method:

- Use factorized condition embeddings for animal ID, object ID, and pair ID, combined into the timestep conditioning path.
- Use classifier-free guidance by randomly dropping conditions during training and mixing conditional/unconditional predictions during sampling.
- Maintain EMA weights and use EMA for sampling.
- Use cosine diffusion schedule if it improves validation samples versus linear schedule.
- Use DDIM-style reduced-step sampling for faster production once quality is acceptable.
- Use basic image augmentations only if they preserve the animal-object semantics and do not harm CLIP-T.

Correctness strategy:

- Validate CSV parsing and category mapping against the 10 animals, 10 objects, and 100 pairs.
- Verify every training image listed in `train.csv` exists and is 64x64 RGB.
- Verify every generated filename exactly matches `generate.csv`.
- Verify `generated_images/` contains exactly 2,000 PNG files at 64x64.
- Use deterministic seeds for generation runs recorded in `README.md`.

Performance strategy:

- Start with a model that fits a single RTX 4070-class GPU, then scale width or residual depth only if training is stable.
- Use mixed precision if available and reproducible enough for the environment.
- Use gradient accumulation or microbatching if batch size is limited by VRAM.
- Track training loss, sample grids per condition, and validation output-contract checks before spending full training time.
- Prefer quality improvements that do not add new external dependencies.

## Alternatives Considered

- Conditional GAN: simpler sampling and no iterative denoising, but higher training instability and mode-collapse risk on a small dataset.
- VAE-only generator: simpler training, but likely weaker sample quality and diversity for FID.
- Latent diffusion with pretrained VAE: assignment permits pretrained VAE as auxiliary, but this adds ambiguity around allowed usage and reproducibility.
- CLIP-conditioned text embeddings: assignment permits pretrained CLIP as auxiliary, but fixed class/pair embeddings are simpler and avoid needing TA clarification.
- Public diffusion package or pretrained checkpoint: rejected because the assignment requires self-implemented training flow and no pretrained generative model weights.

## Module Candidates

- `scripts/train.py`: training entry point for the scratch conditional diffusion model.
- `scripts/generate.py`: generation entry point that reads `dataset/generate.csv` and writes `generated_images/`.
- `scripts/validate_outputs.py`: lightweight output contract checker for 2,000 PNGs, filenames, and dimensions.
- `scripts/package_submission.py`: optional packaging helper for the E3/Codabench archive.
- `scripts/brainrot_data.py`: dataset loading, CSV parsing, category mapping, and image transforms.
- `scripts/model.py`: scratch U-Net and condition/timestep embedding modules.
- `scripts/diffusion.py`: diffusion schedule, training loss, and sampling routines.
- `requirements.txt`: dependency manifest.
- `README.md`: reproducible training and generation instructions.

These are candidates, not final file boundaries. High-level design should keep the file count small unless implementation complexity requires separation.

## Milestones

1. Repository setup:
   - Add dependency manifest.
   - Add dataset inspection and output validation script.
   - Document expected commands.

2. Baseline training path:
   - Implement scratch dataset loader, U-Net denoiser, diffusion loss, and training loop.
   - Train a small pair-conditioned model.
   - Save `model.pth` and a sample grid.

3. Baseline generation path:
   - Generate 2,000 images from `dataset/generate.csv`.
   - Validate filenames, count, format, and dimensions.
   - Produce a first Codabench submission if quality is nontrivial.

4. Quality improvements:
   - Add EMA sampling.
   - Add classifier-free guidance.
   - Tune schedule, guidance scale, sampling steps, and model size.
   - Compare using Codabench and any locally available scorer components.

5. Final reproducibility:
   - Freeze commands and seeds in `README.md`.
   - Ensure clean regeneration from `model.pth`.
   - Package final E3 and Codabench deliverables.

## Validation Plan

- Add a smoke check that loads a tiny batch from `dataset/train.csv`, runs one forward training loss, and confirms gradients flow.
- Add an output checker that fails unless:
  - `generated_images/` has exactly 2,000 files.
  - All filenames match `dataset/generate.csv`.
  - Every file is PNG.
  - Every image is 64x64 RGB.
- Add a generation smoke check that produces a small temporary batch for selected conditions without overwriting final outputs.
- Use the script-compatible scorer command only after the missing reference files and 2,000-vs-3,000 image-count question are resolved.
- Use Codabench as the authoritative metric source for FID and CLIP-T.
- Target at least the first grading thresholds as the first quality goal:
  - FID `<= 90.0142`.
  - CLIP-T `>= 0.2170`.

## Risks and Tradeoffs

- The dataset is small, so overfitting and low diversity are likely risks.
- Strong classifier-free guidance may improve CLIP-T while hurting diversity and FID.
- Larger models may improve quality but increase training time and VRAM pressure.
- Local scorer files are incomplete for full CLIP scoring in the current checkout.
- The assignment requires 2,000 generated images, while local scorer config references 3,000; this must not compromise the official output contract.
- Avoiding pretrained CLIP/VAE in the generator path is simpler and assignment-safe, but may reduce semantic alignment compared with CLIP-assisted conditioning.

## Assumptions

- PyTorch is acceptable for implementation because the repository scorer already uses PyTorch and the assignment requires Python-style deliverables.
- Initial implementation will use only the provided Brainrot Dataset.
- Public repositories will be used only to understand standard DDPM concepts and will not be copied into the submission.
- The final official generation set is `dataset/generate.csv` with 2,000 rows.
- Codabench remains the authoritative final evaluation platform.

## Open Questions

- What student ID should be used for the final `HW6_{student_id}.zip` name?
- Should local scoring be adapted to 2,000 generated images, or should a separate 3,000-image local evaluation fixture be constructed?
- Is TA approval needed before using any pretrained CLIP embedding or pretrained VAE as an auxiliary module?
- What is the target GPU and maximum acceptable training time?
- Should extra data be considered after the baseline, or should the final solution remain dataset-only for reproducibility?
