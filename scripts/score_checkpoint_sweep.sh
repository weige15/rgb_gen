#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PATTERN="${PATTERN:-generated_images_xl_*_ddim250_gs*_test}"
SCORES_TSV="${SCORES_TSV:-runs/checkpoint_sweep_scores_$(date -u +%Y%m%dT%H%M%SZ).tsv}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
SCORES="${SCORES:-fid}"
OVERWRITE="${OVERWRITE:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

extract_score() {
  local score_json="$1"
  python -c 'import json, sys; print(json.load(open(sys.argv[1])).get("FID", ""))' "$score_json"
}

[[ -d scoring_program/input/ref ]] || die "missing scoring reference directory: scoring_program/input/ref"

if [[ -e "$SCORES_TSV" ]] && [[ "$OVERWRITE" != "1" ]]; then
  die "score summary exists: $SCORES_TSV; set OVERWRITE=1 or choose SCORES_TSV=..."
fi

dirs=()
while IFS= read -r image_dir; do
  dirs+=("$image_dir")
done < <(find . -maxdepth 1 -type d -name "$PATTERN" -printf "%f\n" | sort -V)

(( ${#dirs[@]} > 0 )) || die "no generated image directories matched PATTERN=$PATTERN"

mkdir -p "$(dirname "$SCORES_TSV")"
printf "image_dir\tnum_images\tFID\tscore_json\n" > "$SCORES_TSV"

for image_dir in "${dirs[@]}"; do
  n="$(find "$image_dir" -maxdepth 1 -type f -name "*.png" | wc -l)"
  n="${n//[[:space:]]/}"
  if [[ "$n" == "0" ]]; then
    echo "[skip] $image_dir has no PNG files"
    continue
  fi

  safe_name="${image_dir//[^A-Za-z0-9_.-]/_}"
  root="$(mktemp -d "/tmp/score_${safe_name}.XXXXXX")"

  mkdir -p "$root/input/res"
  cp -a scoring_program/input/ref "$root/input/ref"
  cp "$image_dir"/*.png "$root/input/res/"

  echo "[score] image_dir=$image_dir images=$n output=$root/scores.json"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scoring_program/score.py \
    --input_dir "$root/input" \
    --output_dir "$root" \
    --image_size "$IMAGE_SIZE" \
    --num_images "$n" \
    --scores $SCORES \
    --batch_size "$BATCH_SIZE" \
    --verbose

  fid="$(extract_score "$root/scores.json")"
  printf "%s\t%s\t%s\t%s\n" "$image_dir" "$n" "$fid" "$root/scores.json" >> "$SCORES_TSV"
done

echo "[done] wrote $SCORES_TSV"
