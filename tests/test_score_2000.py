from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

from scripts import score_2000
from scripts.brainrot_data import CSVValidationError


class Score2000Tests(unittest.TestCase):
    def test_missing_fid_reference_statistics_fail_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _args(root, scores=["fid"], ref_mu=root / "missing_mu.npy", ref_sigma=root / "missing_sigma.npy")

            with self.assertRaises(FileNotFoundError):
                score_2000.score(args, expected_count=1)

    def test_missing_prompts_fail_for_clip_t(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv = root / "generate.csv"
            with generate_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "animal", "object"])
                writer.writerow(["000001.png", "fish", "chair"])
            args = _args(root, scores=["clip_t"], generate_csv=generate_csv)

            with self.assertRaises(CSVValidationError):
                score_2000.score(args, expected_count=1)

    def test_output_file_overwrite_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "scores.json"
            output.write_text("{}", encoding="utf-8")
            args = _args(root, scores=["clip_t"], output=output)

            with self.assertRaises(FileExistsError):
                score_2000.score(args, expected_count=1)

    def test_submission_validation_enforces_exact_2000_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv = _generate_csv(root / "generate.csv")
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (64, 64), color=(1, 2, 3)).save(image_dir / "000001.png", format="PNG")
            args = _args(root, scores=["clip_t"], generate_csv=generate_csv)
            args.image_dir = str(image_dir)

            with self.assertRaisesRegex(ValueError, "Expected 2000 filenames"):
                score_2000.score(args)

    def test_clip_t_preflight_does_not_require_raw_validation_images_or_reference_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _args(root, scores=["clip_t"])

            with self.assertRaisesRegex(ValueError, "Image directory"):
                score_2000.score(args, expected_count=1)

    def test_validation_failure_does_not_write_unrun_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "scores.json"
            args = _args(root, scores=["clip_t"], output=output)

            with self.assertRaises(ValueError):
                score_2000.score(args, expected_count=1)

        self.assertFalse(output.exists())

    def test_training_and_generation_do_not_import_scoring_adapter(self) -> None:
        train_source = Path("scripts/train.py").read_text(encoding="utf-8")
        generate_source = Path("scripts/generate.py").read_text(encoding="utf-8")

        self.assertNotIn("score_2000", train_source)
        self.assertNotIn("score_2000", generate_source)


def _args(
    root: Path,
    scores: list[str],
    generate_csv: Path | None = None,
    ref_mu: Path | None = None,
    ref_sigma: Path | None = None,
    output: Path | None = None,
) -> Namespace:
    return Namespace(
        image_dir=str(root / "images"),
        generate_csv=str(generate_csv or _generate_csv(root / "generate.csv")),
        ref_mu=str(ref_mu or root / "ref_mu.npy"),
        ref_sigma=str(ref_sigma or root / "ref_sigma.npy"),
        scores=scores,
        output=str(output or root / "scores.json"),
        device="cpu",
        batch_size=2,
        num_workers=0,
        overwrite=False,
    )


def _generate_csv(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "animal", "object", "prompt"])
        writer.writerow(["000001.png", "fish", "chair", "a fish and a chair"])
    return path


if __name__ == "__main__":
    unittest.main()
