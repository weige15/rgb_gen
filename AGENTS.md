# AGENTS.md

## Project Context

- Repository name: `rgb_gen` per `README.md`.
- Purpose: image-generation project with tracked training/generation CSVs, PNG training images, and a local scoring program.
- Main verified code: `scoring_program/score.py` plus project scripts under `scripts/`.
- Language/toolchain: Python. Verified runtime imports include `argparse`, `glob`, `json`, `os`, `numpy`, `open_clip`, `torch`, `PIL`, `torchvision`, `tqdm`, and `scipy`.
- Package manager: `pip` per user instruction. `requirements.txt` is present.

## Repository Rules

- Start each task with read-only discovery before editing.
- Treat large tracked dataset and scoring assets as source data unless the user explicitly asks to change them.
- Do not overwrite generated images, checkpoints, run logs, model weights, or score files without user approval.
- Do not add secrets, tokens, private credentials, or copied environment values to files.
- Keep future edits scoped to the user's requested task.

## Read-Only Discovery Commands

- `pwd`
- `git status --short --branch`
- `git branch --show-current`
- `git worktree list`
- `rg --files`
- `sed -n '1,220p' README.md`
- `sed -n '1,220p' .gitignore`
- `sed -n '1,260p' scoring_program/score.py`
- `sed -n '1,220p' scoring_manual.txt`

## Commands Requiring Permission

- Creating, editing, moving, or deleting files.
- Running install, build, test, lint, format, migration, generation, scoring, training, or server commands.
- Creating, deleting, or changing git branches.
- Committing, pushing, rebasing, merging, stashing, or applying patches.
- Writing to `generated_images/`, `checkpoints/`, `runs/`, `scores.json`, or `scoring_program/scores.json`.

## Forbidden Commands

Forbidden unless the user explicitly requests the exact action:

- `rm -rf`
- `git reset --hard`
- `git clean -fd`
- `git checkout -- .`
- `git restore .`
- `git push --force`
- `git push --force-with-lease`
- `chmod -R`
- `chown -R`
- `sudo`

## Build, Test, and Quality Gates

- Package install command: `python -m pip install -r requirements.txt` after a `requirements.txt` file exists.
- Recommended lightweight build/syntax check: `python -m compileall scoring_program`
- Recommended test command: `python -m unittest discover`
- Recommended lint command: `python -m ruff check .` after installing `ruff`.
- Recommended format command: `python -m ruff format .` after installing `ruff`.
- Recommended type-check command: `python -m mypy scoring_program` after installing `mypy`.
- `requirements.txt` and a `unittest` suite are present. No lint config, format config, or type-check config has been verified yet.
- Verified local scoring setup is documented in `scoring_manual.txt`.
- Verified scoring script path: `scoring_program/score.py`.
- Verified project test command: `python -m unittest discover`.
- Verified project syntax check: `python -m compileall scoring_program scripts`.
- Verified scoring script requires CUDA by default via `torch.device("cuda:0")`.
- Verified scoring command shape from `score.py`:

```bash
cd scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 64 --num_images 3000 --test_json test.json --scores fid clip_t clip_i --verbose
```

- `scoring_program/metadata` also records this platform-style command:

```bash
python3 score.py --input_dir $input --output_dir $output --config config.json
```

## Documentation Rules

- Keep setup and workflow facts in documentation only when verified from files or provided by the user.
- Mark missing project facts as `Unknown` rather than guessing.
- Update this file when adding or verifying build, test, lint, format, dependency, or scoring workflows.
- If dependencies are stabilized, add a `requirements.txt` compatible with `pip`.
- Keep command examples non-destructive and avoid default workflows that overwrite model outputs or scores.

## Coding Rules

- Prefer existing project layout: `dataset/` for data and `scoring_program/` for evaluation code.
- Use Python standard library APIs where they are sufficient.
- Preserve input image dimensions and filename expectations when working with scoring data.
- Avoid broad refactors around `scoring_program/score.py` unless the user asks for scoring changes.
- Validate file paths and generated outputs before writing to tracked or ignored result locations.

## Git and Commit Rules

- Current branch at setup: `main`.
- Worktree at setup: `/home/kuotzuwei15/GenAI/rgb`.
- Do not commit, push, branch, merge, rebase, or stash without user approval.
- Before editing, check `git status --short --branch` and avoid reverting user changes.
- If unrelated changes are present, leave them untouched.

## Uncertainty Protocol

- If a dependency, command, dataset rule, or grading requirement is not verified locally, state that it is unknown.
- Ask the user before running commands that write files, download dependencies, start training, or evaluate generated outputs.
- Ask before modifying large datasets, reference files, scoring outputs, or generated artifacts.
