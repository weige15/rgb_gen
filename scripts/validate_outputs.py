from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from scripts.brainrot_data import CSVValidationError, IMAGE_SIZE, load_generation_requests


def validate_outputs(
    generate_csv: str | Path,
    image_dir: str | Path,
    expected_count: int = 2000,
    strict_prompt: bool = False,
    image_size: int = IMAGE_SIZE,
) -> list[str]:
    errors: list[str] = []
    csv_path = Path(generate_csv)
    output_dir = Path(image_dir)

    try:
        expected_ids = _expected_ids(csv_path, strict_prompt)
    except (CSVValidationError, FileNotFoundError) as exc:
        return [str(exc)]

    if len(expected_ids) != expected_count:
        errors.append(
            f"Expected {expected_count} filenames from {csv_path}, found {len(expected_ids)}."
        )

    if not output_dir.exists():
        errors.append(f"Image directory {output_dir} does not exist.")
        return errors
    if not output_dir.is_dir():
        errors.append(f"Image path {output_dir} is not a directory.")
        return errors

    actual_pngs: set[str] = set()
    non_png_files: list[str] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() == ".png":
            actual_pngs.add(path.name)
        else:
            non_png_files.append(path.name)
    non_png_files.sort()
    expected_set = set(expected_ids)
    missing = sorted(expected_set - actual_pngs)
    extra = sorted(actual_pngs - expected_set)

    if non_png_files:
        errors.append(f"Non-PNG files are not allowed: {_preview(non_png_files)}.")
    if missing:
        errors.append(f"Missing expected PNG files: {_preview(missing)}.")
    if extra:
        errors.append(f"Unexpected PNG files: {_preview(extra)}.")

    for image_id in expected_ids:
        image_path = output_dir / image_id
        if not image_path.exists():
            continue
        errors.extend(_inspect_image(image_path, image_size))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated Brainrot PNG outputs.")
    parser.add_argument("--generate_csv", default="dataset/generate.csv")
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--expected_count", type=int, default=2000)
    parser.add_argument("--strict_prompt", action="store_true")
    args = parser.parse_args(argv)

    errors = validate_outputs(
        generate_csv=args.generate_csv,
        image_dir=args.image_dir,
        expected_count=args.expected_count,
        strict_prompt=args.strict_prompt,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"OK: {args.image_dir} contains {args.expected_count} RGB PNG files matching {args.generate_csv}."
    )
    return 0


def _expected_ids(generate_csv: Path, strict_prompt: bool) -> list[str]:
    if strict_prompt:
        return [request.image_id for request in load_generation_requests(generate_csv)]

    try:
        with generate_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise CSVValidationError(f"{generate_csv}: CSV file is empty.")
            if "id" not in reader.fieldnames:
                raise CSVValidationError(f"{generate_csv}: missing required 'id' column.")
            ids: list[str] = []
            seen: set[str] = set()
            for row_index, row in enumerate(reader):
                image_id = row.get("id", "")
                context = f"{generate_csv}:{row_index + 2}"
                if not image_id:
                    raise CSVValidationError(f"{context}: missing required value 'id'.")
                path = Path(image_id)
                if path.name != image_id or path.suffix.lower() != ".png":
                    raise CSVValidationError(f"{context}: id {image_id!r} must be a PNG filename.")
                if image_id in seen:
                    raise CSVValidationError(f"{context}: duplicate id {image_id!r}.")
                seen.add(image_id)
                ids.append(image_id)
    except FileNotFoundError:
        raise

    if not ids:
        raise CSVValidationError(f"{generate_csv}: CSV file has no data rows.")
    return ids


def _inspect_image(image_path: Path, image_size: int) -> list[str]:
    try:
        with Image.open(image_path) as image:
            image.load()
            errors: list[str] = []
            if image.format != "PNG":
                errors.append(f"{image_path}: format is {image.format!r}, expected 'PNG'.")
            if image.size != (image_size, image_size):
                errors.append(
                    f"{image_path}: size is {image.size}, expected {(image_size, image_size)}."
                )
            if image.mode != "RGB":
                errors.append(f"{image_path}: mode is {image.mode!r}, expected 'RGB'.")
            return errors
    except (OSError, UnidentifiedImageError) as exc:
        return [f"{image_path}: could not be read as a PNG image: {exc}."]


def _preview(values: Iterable[str], limit: int = 10) -> str:
    items = list(values)
    suffix = "" if len(items) <= limit else f" ... and {len(items) - limit} more"
    return ", ".join(items[:limit]) + suffix


if __name__ == "__main__":
    raise SystemExit(main())
