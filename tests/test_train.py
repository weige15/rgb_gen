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
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("scaler_state_dict", checkpoint)
            self.assertIn("diffusion_config", checkpoint)
            self.assertEqual(checkpoint["seed"], 123)

            log_lines = (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(log_lines), 2)
            events = [json.loads(line) for line in log_lines]
            self.assertTrue(any(event["event"] == "step" and event["loss"] is not None for event in events))
            step_event = next(event for event in events if event["event"] == "step")
            required_fields = {"timestamp", "seed", "device", "elapsed_seconds", "eta_seconds", "step", "output_model"}
            self.assertTrue(required_fields.issubset(step_event))
            self.assertEqual(step_event["eta_seconds"], 0.0)
            save_event = next(event for event in events if event["event"] == "save")
            self.assertEqual(save_event["eta_seconds"], 0.0)
            self.assertFalse((root / "generated_images").exists())

    def test_resume_from_old_checkpoint_continues_step_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv, image_dir = _training_fixture(root)
            run_dir = root / "run"
            first_model = root / "model_step1.pth"

            exit_code = train.main(
                _smoke_args(train_csv, image_dir, first_model, run_dir)
                + ["--save_every_steps", "1", "--ema"]
            )

            self.assertEqual(exit_code, 0)
            old_checkpoint_path = root / "old_step_000001.pth"
            old_checkpoint = torch.load(run_dir / "checkpoints" / "step_000001.pth", map_location="cpu")
            old_checkpoint.pop("optimizer_state_dict")
            old_checkpoint.pop("scaler_state_dict")
            torch.save(old_checkpoint, old_checkpoint_path)

            second_model = root / "model_step2.pth"
            exit_code = train.main(
                _smoke_args(train_csv, image_dir, second_model, run_dir, max_steps=2)
                + ["--resume_from", str(old_checkpoint_path), "--save_every_steps", "1", "--ema"]
            )

            self.assertEqual(exit_code, 0)
            resumed = torch.load(second_model, map_location="cpu")
            self.assertEqual(resumed["train_step"], 2)
            self.assertIn("optimizer_state_dict", resumed)
            self.assertIn("scaler_state_dict", resumed)
            self.assertTrue((run_dir / "checkpoints" / "step_000002.pth").exists())

            events = [
                json.loads(line)
                for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            resume_event = next(event for event in events if event["event"] == "resume")
            self.assertEqual(resume_event["step"], 1)
            self.assertEqual(resume_event["resume_from"], str(old_checkpoint_path))

    def test_progress_timing_estimates_remaining_steps(self) -> None:
        event = {"elapsed_seconds": None, "eta_seconds": None}

        with mock.patch("scripts.train.time.monotonic", return_value=14.0):
            train._add_progress_timing(event, started=10.0, step=2, expected_steps=5)

        self.assertEqual(event["elapsed_seconds"], 4.0)
        self.assertEqual(event["eta_seconds"], 6.0)

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

    def test_attention_args_are_saved_in_model_config(self) -> None:
        args = train.parse_args(["--attention_resolutions", "16,8", "--attention_heads", "2"])

        model_config, _ = train._training_configs(args, None)

        self.assertEqual(model_config.attention_resolutions, (16, 8))
        self.assertEqual(model_config.attention_heads, 2)

    def test_conditioning_arg_is_saved_in_model_config(self) -> None:
        args = train.parse_args(["--conditioning", "film"])

        model_config, _ = train._training_configs(args, None)

        self.assertEqual(model_config.conditioning, "film")


def _smoke_args(
    train_csv: Path,
    image_dir: Path,
    output_model: Path,
    run_dir: Path,
    max_steps: int = 1,
) -> list[str]:
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
        str(max_steps),
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
