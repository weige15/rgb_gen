from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from scripts import train


class TrainOrchestratorTests(unittest.TestCase):
    def test_one_step_cpu_smoke_writes_checkpoint_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv, image_dir = _training_fixture(root)
            output_model = root / "model.pth"
            run_dir = root / "run"

            with mock.patch("torch.load", side_effect=AssertionError("training must not load checkpoints")):
                exit_code = train.main(
                    _smoke_args(train_csv, image_dir, output_model, run_dir)
                )

            self.assertEqual(exit_code, 0)
            checkpoint = torch.load(output_model, map_location="cpu")
            self.assertEqual(checkpoint["format_version"], 1)
            self.assertFalse(checkpoint["uses_pretrained_generator_weights"])
            self.assertEqual(checkpoint["train_step"], 1)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("diffusion_config", checkpoint)
            self.assertEqual(checkpoint["seed"], 123)

            log_lines = (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(log_lines), 2)
            events = [json.loads(line) for line in log_lines]
            self.assertTrue(any(event["event"] == "step" and event["loss"] is not None for event in events))
            step_event = next(event for event in events if event["event"] == "step")
            required_fields = {"timestamp", "seed", "device", "elapsed_seconds", "eta_seconds", "step", "output_model"}
            self.assertTrue(required_fields.issubset(step_event))
            self.assertFalse((root / "generated_images").exists())

    def test_existing_output_model_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv, image_dir = _training_fixture(root)
            output_model = root / "model.pth"
            output_model.write_bytes(b"existing")

            with redirect_stdout(StringIO()):
                exit_code = train.main(
                    _smoke_args(train_csv, image_dir, output_model, root / "run")
                )

        self.assertEqual(exit_code, 1)

    def test_parse_device_ids_rejects_more_than_four_gpus(self) -> None:
        with self.assertRaises(ValueError):
            train.parse_device_ids("0,1,2,3,4")

    def test_parse_device_ids_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            train.parse_device_ids("1,1")

    def test_resolve_device_allows_cpu_smoke_without_cuda(self) -> None:
        device = train.resolve_device([0, 1, 2, 3], cpu_smoke=True, local_rank=0, world_size=1)

        self.assertEqual(device.type, "cpu")

    def test_train_source_does_not_hardcode_cuda_zero(self) -> None:
        source = Path("scripts/train.py").read_text(encoding="utf-8")

        self.assertNotIn("cuda:0", source)


def _smoke_args(train_csv: Path, image_dir: Path, output_model: Path, run_dir: Path) -> list[str]:
    return [
        "--train_csv",
        str(train_csv),
        "--image_dir",
        str(image_dir),
        "--output_model",
        str(output_model),
        "--run_dir",
        str(run_dir),
        "--seed",
        "123",
        "--epochs",
        "1",
        "--max_steps",
        "1",
        "--batch_size",
        "2",
        "--cpu_smoke",
        "--base_channels",
        "4",
        "--channel_multipliers",
        "1",
        "--embedding_dim",
        "16",
        "--train_timesteps",
        "4",
        "--sampling_steps",
        "4",
    ]


def _training_fixture(root: Path) -> tuple[Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir()
    _write_png(image_dir / "000001.png")
    _write_png(image_dir / "000002.png")
    train_csv = root / "train.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "animal", "object"])
        writer.writerow(["000001.png", "fish", "chair"])
        writer.writerow(["000002.png", "cat", "toilet"])
    return train_csv, image_dir


def _write_png(path: Path) -> None:
    Image.new("RGB", (64, 64), color=(120, 80, 40)).save(path, format="PNG")


if __name__ == "__main__":
    unittest.main()
