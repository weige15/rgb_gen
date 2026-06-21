from __future__ import annotations

import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest import mock

import torch

from scripts import generate
from scripts.brainrot_data import ANIMALS, OBJECTS, load_generation_requests
from scripts.diffusion import DiffusionConfig
from scripts.model import ConditionalUNet, UNetConfig
from scripts.validate_outputs import validate_outputs


class GenerateTests(unittest.TestCase):
    def test_temporary_generation_writes_requested_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root / "model.pth")
            generate_csv = _write_generate_csv(root / "generate.csv")
            output_dir = root / "out"
            run_dir = root / "run"

            with redirect_stdout(StringIO()):
                exit_code = generate.main(
                    _generate_args(checkpoint, generate_csv, output_dir, run_dir)
                )

            errors = validate_outputs(generate_csv, output_dir, expected_count=2, strict_prompt=True)

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])

    def test_same_seed_repeats_outputs_on_same_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root / "model.pth")
            generate_csv = _write_generate_csv(root / "generate.csv")
            first_dir = root / "first"
            second_dir = root / "second"

            with redirect_stdout(StringIO()):
                first_code = generate.main(
                    _generate_args(checkpoint, generate_csv, first_dir, root / "run1")
                )
                second_code = generate.main(
                    _generate_args(checkpoint, generate_csv, second_dir, root / "run2")
                )

            first_bytes = (first_dir / "000001.png").read_bytes()
            second_bytes = (second_dir / "000001.png").read_bytes()

        self.assertEqual(first_code, 0)
        self.assertEqual(second_code, 0)
        self.assertEqual(first_bytes, second_bytes)

    def test_existing_output_dir_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root / "model.pth")
            generate_csv = _write_generate_csv(root / "generate.csv")
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "old.png").write_bytes(b"old")

            with redirect_stdout(StringIO()):
                exit_code = generate.main(
                    _generate_args(checkpoint, generate_csv, output_dir, root / "run")
                )

        self.assertEqual(exit_code, 1)

    def test_partition_requests_covers_every_request_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generate_csv = _write_generate_csv(Path(tmp) / "generate.csv", count=5)
            requests = load_generation_requests(generate_csv)

            partitions = [generate.partition_requests(requests, rank, 3) for rank in range(3)]

        image_ids = [request.image_id for partition in partitions for request in partition]
        self.assertEqual(sorted(image_ids), [f"{index:06d}.png" for index in range(1, 6)])
        self.assertEqual(len(image_ids), len(set(image_ids)))

    def test_checkpoint_category_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root / "model.pth", animals=["bad", *ANIMALS[1:]])
            generate_csv = _write_generate_csv(root / "generate.csv")

            with redirect_stdout(StringIO()):
                exit_code = generate.main(
                    _generate_args(checkpoint, generate_csv, root / "out", root / "run")
                )

        self.assertEqual(exit_code, 1)

    def test_generation_does_not_call_pretrained_model_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root / "model.pth")
            generate_csv = _write_generate_csv(root / "generate.csv")

            with mock.patch("torch.hub.load", side_effect=AssertionError("pretrained load not allowed")):
                with redirect_stdout(StringIO()):
                    exit_code = generate.main(
                        _generate_args(checkpoint, generate_csv, root / "out", root / "run")
                    )

        self.assertEqual(exit_code, 0)

    def test_generate_source_does_not_import_scoring_or_pretrained_packages(self) -> None:
        source = Path("scripts/generate.py").read_text(encoding="utf-8")

        self.assertNotIn("open_clip", source)
        self.assertNotIn("torchvision", source)
        self.assertNotIn("scoring_program", source)


def _generate_args(model: Path, generate_csv: Path, output_dir: Path, run_dir: Path) -> list[str]:
    return [
        "--model",
        str(model),
        "--generate_csv",
        str(generate_csv),
        "--output_dir",
        str(output_dir),
        "--run_dir",
        str(run_dir),
        "--seed",
        "99",
        "--sampling_steps",
        "2",
        "--batch_size",
        "1",
        "--limit",
        "2",
        "--cpu_smoke",
        "--quiet",
    ]


def _write_checkpoint(path: Path, animals: list[str] | None = None) -> Path:
    model_config = UNetConfig(
        base_channels=4,
        channel_multipliers=(1,),
        embedding_dim=16,
        residual_blocks=1,
        dropout=0.0,
    )
    model = ConditionalUNet(model_config)
    diffusion_config = DiffusionConfig(train_timesteps=4, sampling_steps=2, cfg_dropout=0.0)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None,
            "model_config": asdict(model_config),
            "diffusion_config": asdict(diffusion_config),
            "animals": animals or list(ANIMALS),
            "objects": list(OBJECTS),
            "seed": 123,
            "train_step": 1,
            "uses_pretrained_generator_weights": False,
        },
        path,
    )
    return path


def _write_generate_csv(path: Path, count: int = 2) -> Path:
    rows = [
        ["000001.png", "fish", "chair", "a fish and a chair"],
        ["000002.png", "cat", "toilet", "a cat and a toilet"],
        ["000003.png", "dog", "car", "a dog and a car"],
        ["000004.png", "bird", "drum", "a bird and a drum"],
        ["000005.png", "shark", "sneaker", "a shark and a sneaker"],
    ][:count]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "animal", "object", "prompt"])
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    unittest.main()
