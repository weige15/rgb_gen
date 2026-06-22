#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/train_xl_attention_film_100k/checkpoints}"
GENERATE_CSV="${GENERATE_CSV:-dataset/generate.csv}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-generated_images_xl}"
RUN_PREFIX="${RUN_PREFIX:-runs/generate_xl}"
SEED="${SEED:-1234}"
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAMPLER="${SAMPLER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-250}"
DDIM_ETA="${DDIM_ETA:-0.0}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-3.0}"
WEIGHTS="${WEIGHTS:-raw ema}"
STEPS="${STEPS:-}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

has_files() {
  [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

normalize_checkpoint_name() {
  local value="$1"
  value="${value##*/}"
  value="${value%.pth}"
  value="${value#step_}"

  if [[ "$value" =~ ^[0-9]+k$ ]]; then
    printf "step_%06d.pth" "$((10#${value%k} * 1000))"
  elif [[ "$value" =~ ^[0-9]+$ ]]; then
    printf "step_%06d.pth" "$((10#$value))"
  else
    printf "%s" "$1"
  fi
}

checkpoint_label() {
  local name="${1##*/}"
  local value="${name#step_}"
  value="${value%.pth}"
  local numeric="$((10#$value))"
  if (( numeric % 1000 == 0 )); then
    printf "%dk" "$((numeric / 1000))"
  else
    printf "%d" "$numeric"
  fi
}

print_command() {
  printf "CUDA_VISIBLE_DEVICES=%q" "$CUDA_DEVICE"
  printf " %q" "$@"
  printf "\n"
}

[[ -d "$CHECKPOINT_DIR" ]] || die "checkpoint directory not found: $CHECKPOINT_DIR"
[[ -f "$GENERATE_CSV" ]] || die "generation CSV not found: $GENERATE_CSV"

read -r -a guidance_scales <<< "$GUIDANCE_SCALES"
read -r -a weights <<< "$WEIGHTS"

(( ${#guidance_scales[@]} > 0 )) || die "GUIDANCE_SCALES is empty"
(( ${#weights[@]} > 0 )) || die "WEIGHTS is empty"

checkpoint_paths=()
if [[ -n "$STEPS" ]]; then
  read -r -a requested_steps <<< "$STEPS"
  for step in "${requested_steps[@]}"; do
    checkpoint_name="$(normalize_checkpoint_name "$step")"
    checkpoint_path="$CHECKPOINT_DIR/$checkpoint_name"
    [[ -f "$checkpoint_path" ]] || die "checkpoint not found: $checkpoint_path"
    checkpoint_paths+=("$checkpoint_path")
  done
else
  while IFS= read -r checkpoint_path; do
    checkpoint_paths+=("$checkpoint_path")
  done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "step_*.pth" | sort -V)
fi

(( ${#checkpoint_paths[@]} > 0 )) || die "no checkpoints found in $CHECKPOINT_DIR"

for checkpoint_path in "${checkpoint_paths[@]}"; do
  step_label="$(checkpoint_label "$checkpoint_path")"
  sampler_tag="${SAMPLER}${SAMPLING_STEPS}"

  for weight in "${weights[@]}"; do
    case "$weight" in
      raw|ema) ;;
      *) die "unsupported weight mode '$weight'; use raw and/or ema" ;;
    esac

    for guidance_scale in "${guidance_scales[@]}"; do
      output_dir="${OUTPUT_PREFIX}_${step_label}_${weight}_${sampler_tag}_gs${guidance_scale}_test"
      run_dir="${RUN_PREFIX}_${step_label}_${weight}_${sampler_tag}_gs${guidance_scale}_test"

      if [[ "$OVERWRITE" != "1" ]] && has_files "$output_dir"; then
        echo "[skip] $output_dir already has files; set OVERWRITE=1 to regenerate"
        continue
      fi

      cmd=(
        python scripts/generate.py
        --model "$checkpoint_path"
        --generate_csv "$GENERATE_CSV"
        --output_dir "$output_dir"
        --run_dir "$run_dir"
        --seed "$SEED"
        --batch_size "$BATCH_SIZE"
        --sampler "$SAMPLER"
        --sampling_steps "$SAMPLING_STEPS"
        --guidance_scale "$guidance_scale"
      )

      if [[ "$SAMPLER" == "ddim" ]]; then
        cmd+=(--ddim_eta "$DDIM_ETA")
      fi
      if [[ -n "$LIMIT" ]] && [[ "$LIMIT" != "0" ]]; then
        cmd+=(--limit "$LIMIT")
      fi
      if [[ "$weight" == "ema" ]]; then
        cmd+=(--use_ema)
      fi
      if [[ "$OVERWRITE" == "1" ]]; then
        cmd+=(--overwrite)
      fi

      echo "[generate] checkpoint=$checkpoint_path weight=$weight guidance=$guidance_scale output=$output_dir"
      print_command "${cmd[@]}"
      if [[ "$DRY_RUN" != "1" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
      fi
    done
  done
done
