from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.brainrot_data import (
    CSVValidationError,
    ImageValidationError,
    load_extra_records,
    validate_extra_manifest,
)


class ExtraDataTests(unittest.TestCase):
    def test_no_manifest_returns_no_records(self) -> None:
        self.assertEqual(load_extra_records(None), [])
        self.assertEqual(validate_extra_manifest(None), [])

    def test_valid_manifest_returns_training_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            _write_manifest(manifest, [_valid_row(image_path.name, image_path)])

            records = load_extra_records(manifest)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_id, "extra.png")
        self.assertEqual(records[0].condition.animal, "fish")
        self.assertEqual(records[0].condition.object, "chair")
        self.assertEqual(records[0].source, "curated-local-set")

    def test_missing_manifest_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            row = _valid_row(image_path.name, image_path)
            columns = [column for column in _COLUMNS if column != "image_hash"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow({key: value for key, value in row.items() if key in columns})

            with self.assertRaises(CSVValidationError):
                load_extra_records(manifest)

    def test_missing_provenance_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            row = _valid_row(image_path.name, image_path)
            row["license_or_allowance"] = ""
            _write_manifest(manifest, [row])

            with self.assertRaises(CSVValidationError):
                load_extra_records(manifest)

    def test_unknown_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            row = _valid_row(image_path.name, image_path)
            row["animal"] = "lion"
            _write_manifest(manifest, [row])

            with self.assertRaises(CSVValidationError):
                load_extra_records(manifest)

    def test_missing_image_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "extra.csv"
            row = _valid_row("missing.png", root / "missing.png")
            _write_manifest(manifest, [row])

            with self.assertRaises(FileNotFoundError):
                load_extra_records(manifest)

    def test_wrong_image_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path, size=(64, 63))
            manifest = root / "extra.csv"
            _write_manifest(manifest, [_valid_row(image_path.name, image_path)])

            with self.assertRaises(ImageValidationError):
                load_extra_records(manifest)

    def test_non_rgb_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            Image.new("L", (64, 64), color=128).save(image_path, format="PNG")
            manifest = root / "extra.csv"
            _write_manifest(manifest, [_valid_row(image_path.name, image_path)])

            with self.assertRaises(ImageValidationError):
                load_extra_records(manifest)

    def test_pretrained_generated_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            row = _valid_row(image_path.name, image_path)
            row["preprocessing_notes"] = "stable diffusion generated sample"
            _write_manifest(manifest, [row])

            with self.assertRaises(CSVValidationError):
                load_extra_records(manifest)

    def test_pretrained_generated_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            row = _valid_row(image_path.name, image_path)
            row["label_author_or_method"] = "OpenCLIP labeled"
            _write_manifest(manifest, [row])

            with self.assertRaises(CSVValidationError):
                load_extra_records(manifest)

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "extra.png"
            _write_png(image_path)
            manifest = root / "extra.csv"
            row = _valid_row(image_path.name, image_path)
            row["image_hash"] = "0" * 64
            _write_manifest(manifest, [row])

            with self.assertRaises(CSVValidationError):
                load_extra_records(manifest)


_COLUMNS = [
    "image_path",
    "animal",
    "object",
    "source",
    "retrieval_date",
    "license_or_allowance",
    "label_author_or_method",
    "image_hash",
    "preprocessing_notes",
]


def _valid_row(image_path_value: str, image_path: Path) -> dict[str, str]:
    return {
        "image_path": image_path_value,
        "animal": "fish",
        "object": "chair",
        "source": "curated-local-set",
        "retrieval_date": "2026-06-21",
        "license_or_allowance": "course assignment allowed local data",
        "label_author_or_method": "manual human label",
        "image_hash": hashlib.sha256(image_path.read_bytes()).hexdigest()
        if image_path.exists()
        else "0" * 64,
        "preprocessing_notes": "none",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, color=(210, 64, 92)).save(path, format="PNG")


if __name__ == "__main__":
    unittest.main()
