from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from scripts.brainrot_data import (
    ANIMALS,
    OBJECTS,
    BrainrotDataset,
    CSVValidationError,
    ImageValidationError,
    build_condition,
    load_generation_requests,
    load_train_records,
    validate_prompt,
)


class BrainrotDataTests(unittest.TestCase):
    def test_condition_mapping_uses_assignment_order(self) -> None:
        condition = build_condition("fish", "chair")

        self.assertEqual(condition.animal_id, ANIMALS.index("fish"))
        self.assertEqual(condition.object_id, OBJECTS.index("chair"))
        self.assertEqual(condition.pair_id, 89)

    def test_load_train_records_and_dataset_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            _write_png(image_dir / "000001.png")
            train_csv = root / "train.csv"
            _write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])

            records = load_train_records(train_csv, image_dir)
            dataset = BrainrotDataset(records)
            image, condition_ids = dataset[0]

        self.assertEqual(records[0].image_id, "000001.png")
        self.assertEqual(records[0].condition.animal, "fish")
        self.assertEqual(records[0].condition.object, "chair")
        self.assertEqual(records[0].source, "dataset")
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(image.dtype, torch.float32)
        self.assertGreaterEqual(float(image.min()), -1.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertEqual(condition_ids, {"animal_id": 8, "object_id": 9, "pair_id": 89})

    def test_optional_augmentation_preserves_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            _write_png(image_dir / "000001.png")
            train_csv = root / "train.csv"
            _write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])

            records = load_train_records(train_csv, image_dir)
            image, condition_ids = BrainrotDataset(records, augment=True)[0]

        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(condition_ids, {"animal_id": 8, "object_id": 9, "pair_id": 89})

    def test_generation_requests_validate_coffee_cup_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generate_csv = Path(tmp) / "generate.csv"
            _write_csv(
                generate_csv,
                ["id", "animal", "object", "prompt"],
                [["000001.png", "shark", "coffee cup", "a shark and a coffee cup"]],
            )

            requests = load_generation_requests(generate_csv)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].image_id, "000001.png")
        self.assertEqual(requests[0].prompt, "a shark and a coffee cup")
        self.assertEqual(requests[0].condition.object_id, 2)
        self.assertEqual(requests[0].row_index, 0)

    def test_prompt_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CSVValidationError):
            validate_prompt("shark", "sneaker", "a shark and sneaker")

    def test_duplicate_train_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            _write_png(image_dir / "000001.png")
            train_csv = root / "train.csv"
            _write_csv(
                train_csv,
                ["id", "animal", "object"],
                [
                    ["000001.png", "fish", "chair"],
                    ["000001.png", "cat", "toilet"],
                ],
            )

            with self.assertRaises(CSVValidationError):
                load_train_records(train_csv, image_dir)

    def test_unknown_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            _write_png(image_dir / "000001.png")
            train_csv = root / "train.csv"
            _write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "lion", "chair"]])

            with self.assertRaises(CSVValidationError):
                load_train_records(train_csv, image_dir)

    def test_missing_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            train_csv = root / "train.csv"
            _write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])

            with self.assertRaises(FileNotFoundError):
                load_train_records(train_csv, image_dir)

    def test_wrong_image_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            _write_png(image_dir / "000001.png", size=(63, 64))
            train_csv = root / "train.csv"
            _write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])

            with self.assertRaises(ImageValidationError):
                load_train_records(train_csv, image_dir)

    def test_non_rgb_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            Image.new("L", (64, 64), color=128).save(image_dir / "000001.png", format="PNG")
            train_csv = root / "train.csv"
            _write_csv(train_csv, ["id", "animal", "object"], [["000001.png", "fish", "chair"]])

            with self.assertRaises(ImageValidationError):
                load_train_records(train_csv, image_dir)

    def test_strict_csv_columns_reject_extra_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "trainset"
            image_dir.mkdir()
            _write_png(image_dir / "000001.png")
            train_csv = root / "train.csv"
            _write_csv(
                train_csv,
                ["id", "animal", "object", "extra"],
                [["000001.png", "fish", "chair", "ignored"]],
            )

            with self.assertRaises(CSVValidationError):
                load_train_records(train_csv, image_dir)

    def test_mapping_is_stable_across_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"
            header = ["id", "animal", "object", "prompt"]
            first_rows = [
                ["000001.png", "fish", "chair", "a fish and a chair"],
                ["000002.png", "cat", "toilet", "a cat and a toilet"],
            ]
            second_rows = list(reversed(first_rows))
            _write_csv(first_csv, header, first_rows)
            _write_csv(second_csv, header, second_rows)

            first = {request.image_id: request.condition.pair_id for request in load_generation_requests(first_csv)}
            second = {
                request.image_id: request.condition.pair_id
                for request in load_generation_requests(second_csv)
            }

        self.assertEqual(first, second)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, color=(32, 128, 240)).save(path, format="PNG")


if __name__ == "__main__":
    unittest.main()
