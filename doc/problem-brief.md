# Problem Brief

## Source Documents

- `HW6-Brainrot Image Generation.pdf` - read. Assignment specification for HW6, including objective, dataset, constraints, grading, submission format, and deliverables.
- `scoring_manual.txt` - read. Optional local scoring program setup and command-line usage.

## Assignment Objective

Train a Conditional Image Generation model from scratch to generate Brainrot images conditioned on animal-object pairs. The final submission must include 2,000 generated PNG images at 64x64 resolution, corresponding to the IDs and conditions in `generate.csv`.

The generated images are evaluated against a hidden test set using FID and CLIP-T. FID measures distribution similarity to the test images; CLIP-T measures semantic alignment between each generated image and its text prompt.

## Required Inputs

- Brainrot training dataset with 4,799 images.
- `train.csv`, mapping each training image ID to `animal` and `object` columns:
  - Example schema: `id,animal,object`.
- `generate.csv`, containing 2,000 requested generations:
  - Schema: `id,animal,object,prompt`.
  - Prompt format: `a {animal} and a {object}`.
  - Each animal-object pair appears 20 times.
- Animal categories from the assignment: shark, crocodile, frog, cat, dog, capybara, elephant, bird, fish, monkey.
- Object categories from the assignment: sneaker, airplane, coffee cup, banana, cactus, toilet, pizza, drum, car, chair.
- Optional local evaluation files from Codabench:
  - `scoring_program.zip`.
  - `hw6_reference.zip`.

## Required Outputs

- `generated_images/` containing 2,000 PNG images.
- Each generated image must:
  - Match an ID from `generate.csv`.
  - Use PNG format.
  - Have 64x64 resolution.
  - Correspond to the provided animal-object condition and text prompt.
- E3 submission zip named `HW6_{student_id}.zip`.
- Expected top-level submission folder structure:
  - `generated_images/` with 2,000 PNG images.
  - `scripts/` with training, generation, and related configuration code.
  - `model.pth` with the model weights used for image generation.
  - `README.md` describing training setup, environment setup, and image generation commands.
  - `requirements.txt` listing Python dependencies.
- Optional local scorer output: `scores.json` containing scores such as `CLIP_T`, `CLIP_I`, and `FID`.

## Constraints

- The main Conditional Image Generation model must be trained from scratch.
- The generation model must not use pretrained weights, including pretrained UNet, Transformer, Diffusion Model, or other generative model weights.
- Students must implement the model architecture and training flow themselves, such as backbone, scheduler, loss function, training loop, and sampling procedure.
- High-level generative model packages or ready-made pipelines/training flows, including `diffusers` pipelines, are not allowed as the main generation implementation.
- Pretrained VAE or pretrained CLIP may be used only as auxiliary modules, such as for latent representation, feature extraction, or evaluation.
- Auxiliary pretrained modules must not replace the main generation model and must not directly generate final images.
- Condition design is not restricted. The assignment explicitly allows options such as class embeddings, pretrained CLIP embeddings, or other designs for incorporating animal/object conditions.
- Extra training data outside the Brainrot Dataset is allowed, but the assignment does not guarantee it will improve final evaluation.
- Training flow and final trained model must be reproducible by the teaching assistants.
- The Brainrot Dataset is for this course assignment only and should not be used for other purposes.
- Codabench daily upload limit is 3 submissions.

## Evaluation or Grading Criteria

- Official grading uses Codabench with two metrics:
  - FID: 50% of total grade, lower is better.
  - CLIP-T: 50% of total grade, higher is better.
- FID thresholds:
  - `<= 49.2545`: 100% for the FID portion.
  - `<= 58.0755`: 90% for the FID portion.
  - `<= 75.0642`: 80% for the FID portion.
  - `<= 90.0142`: 70% for the FID portion.
  - Others: 0% for the FID portion.
- CLIP-T thresholds:
  - `>= 0.2703`: 100% for the CLIP-T portion.
  - `>= 0.2618`: 90% for the CLIP-T portion.
  - `>= 0.2536`: 80% for the CLIP-T portion.
  - `>= 0.2170`: 70% for the CLIP-T portion.
  - Others: 0% for the CLIP-T portion.
- CLIP-T is computed with OpenAI CLIP `ViT-B-32-quickgelu`.
- The assignment says `test_mu.npy` and `test_sigma.npy` are provided through Codabench for local FID reference during development.
- Deduction and penalty rules:
  - Late submission: only one week of late submission is allowed; grade is multiplied by 0.7.
  - Incorrect file format: minus 10 points.
  - Plagiarism: copied-from student minus 10 points; copying student receives 0.
  - Non-reproducible results: 0.
  - Violation of assignment restrictions: 0.
  - Not participating in the competition: 0.
- The optional local scorer can be run with:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --score fid clip_t clip_i --verbose
```

## Required Deliverables

- Codabench submission: zip containing the contents of `generated_images/`.
- E3 submission: `HW6_{student_id}.zip`, containing:
  - `generated_images/`.
  - `scripts/`.
  - `model.pth`.
  - `README.md`.
  - `requirements.txt`.
- `README.md` must explain how to use the trained model weights to generate images so teaching assistants can verify and reproduce the results.
- If extra data is used or the model weights exceed the E3 upload limit, the assignment allows providing cloud links in the submission.

## Relevant Methods From Papers

None found in the provided sources.

## Data, Benchmarks, or Test Cases

- Brainrot Dataset contains images built from 10 animal categories and 10 object categories, for 100 total animal-object combinations.
- Training set: 4,799 images.
- Hidden test set: 3,000 images, with 30 images per animal-object pair.
- Required generated set: 2,000 images, with 20 images per animal-object pair.
- The hidden test set is not directly provided and is used only for final evaluation.
- Optional local scoring setup from `scoring_manual.txt` expects this folder structure under `scoring_program/`:
  - `input/ref/test/` with reference images.
  - `input/ref/test.json`.
  - `input/ref/test_mu.npy`.
  - `input/ref/test_sigma.npy`.
  - `input/res/` with generated images using filenames matching reference/test JSON entries.
- The local scorer writes final scores to `scores.json`.

## Implementation Environment

- The submission must include `requirements.txt` for Python packages needed to train and generate.
- The submission must include scripts and configuration needed for training and image generation.
- Baseline models in the assignment are reported as approximately 62.68M parameters and trained on RTX 4070 with 12GB VRAM, but the source does not state this as a required environment.
- The official CLIP-T evaluation uses OpenAI CLIP `ViT-B-32-quickgelu`.
- Codabench is the official scoring platform; E3 is used for final assignment upload.

## Confirmed Facts

- The task is HW6: Brainrot Image Generation.
- The goal is conditional image generation from animal-object pairs.
- The final generated images must be 64x64 PNGs.
- The final generation target is 2,000 images from `generate.csv`.
- The main generation model must be trained from scratch.
- Pretrained generative weights and ready-made high-level generation pipelines are disallowed for the main model.
- Pretrained VAE or CLIP may be used only as auxiliary modules.
- Official grading uses FID and CLIP-T, each worth 50%.
- Result reproducibility is required; non-reproducible results receive 0.
- The optional local scorer setup and command are documented in `scoring_manual.txt`.

## Assumptions

- No additional assumptions are required to create this brief.
- Future implementation planning should assume `generate.csv`, `train.csv`, the image data, and Codabench files are available separately because these files were described by the sources but not included in the named source list.

## Open Questions

- What is the exact official E3 deadline? The PDF text references the assignment deadline but the extracted source does not include a concrete date.
- What student ID should be used in the final zip filename?
- Are `train.csv`, `generate.csv`, training images, and Codabench files already available locally, or must they be downloaded from Codabench?
- The assignment requires 2,000 generated images for `generate.csv`, while the optional local scorer example uses `--num_images 3000`. Confirm which local evaluation setup should be used during development.
- If pretrained CLIP embeddings or a pretrained VAE are used as auxiliary modules, should the planned approach be cleared with the HW6 teaching assistant before implementation?

## Notes for Proposal Generation

- The proposal should preserve the source-grounded constraints around scratch training and restricted pretrained module usage.
- The proposal should not rely on pretrained generative pipelines or hidden test data.
- The implementation plan should include reproducible training, generation scripts, a clear README, and dependency capture in `requirements.txt`.
- The proposal should explicitly decide how to evaluate locally while acknowledging the 2,000-image official submission requirement and the local scorer's 3,000-image example.
- Implementation design should start from the confirmed facts and open questions above.
