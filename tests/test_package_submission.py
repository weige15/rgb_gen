from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

from PIL import Image

from scripts.package_submission import DEFAULT_STUDENT_ID, package_submission


class PackageSubmissionTests(unittest.TestCase):
    def test_package_contains_required_top_level_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _fixture(root)

            result = package_submission(args, expected_count=2)
            names = _zip_names(Path(result["archive"]))

        self.assertEqual(Path(result["archive"]).name, "HW6_314511048.zip")
        self.assertIn("generated_images/000001.png", names)
        self.assertIn("generated_images/000002.png", names)
        self.assertIn("scripts/train.py", names)
        self.assertIn("model.pth", names)
        self.assertIn("README.md", names)
        self.assertIn("requirements.txt", names)
        self.assertNotIn("scripts/runs/debug.txt", names)
        self.assertNotIn("scripts/checkpoints/old.pt", names)
        self.assertNotIn("scripts/__pycache__/ignored.pyc", names)
        self.assertNotIn("scripts/scores.json", names)

    def test_missing_model_fails_before_archive_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _fixture(root)
            Path(args.model).unlink()

            with self.assertRaises(FileNotFoundError):
                package_submission(args, expected_count=2)

            self.assertFalse(Path(args.output).exists())

    def test_validation_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _fixture(root)
            (Path(args.generated_dir) / "000002.png").unlink()

            with self.assertRaisesRegex(ValueError, "Generated outputs failed validation"):
                package_submission(args, expected_count=2)

            self.assertFalse(Path(args.output).exists())

    def test_existing_archive_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _fixture(root)
            Path(args.output).write_bytes(b"old")

            with self.assertRaises(FileExistsError):
                package_submission(args, expected_count=2)

    def test_wrong_archive_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _fixture(root)
            args.output = str(root / "wrong.zip")

            with self.assertRaises(ValueError):
                package_submission(args, expected_count=2)


def _fixture(root: Path) -> Namespace:
    generated_dir = root / "generated"
    generated_dir.mkdir()
    _write_png(generated_dir / "000001.png")
    _write_png(generated_dir / "000002.png")
    generate_csv = root / "generate.csv"
    with generate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "animal", "object", "prompt"])
        writer.writerow(["000001.png", "fish", "chair", "a fish and a chair"])
        writer.writerow(["000002.png", "cat", "toilet", "a cat and a toilet"])

    scripts_dir = root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "train.py").write_text("print('train')\n", encoding="utf-8")
    (scripts_dir / "scores.json").write_text("{}", encoding="utf-8")
    (scripts_dir / "runs").mkdir()
    (scripts_dir / "runs" / "debug.txt").write_text("skip", encoding="utf-8")
    (scripts_dir / "checkpoints").mkdir()
    (scripts_dir / "checkpoints" / "old.pt").write_bytes(b"skip")
    (scripts_dir / "__pycache__").mkdir()
    (scripts_dir / "__pycache__" / "ignored.pyc").write_bytes(b"skip")

    model = root / "model.pth"
    model.write_bytes(b"model")
    readme = root / "README.md"
    readme.write_text("# test\n", encoding="utf-8")
    requirements = root / "requirements.txt"
    requirements.write_text("torch\n", encoding="utf-8")
    return Namespace(
        student_id=DEFAULT_STUDENT_ID,
        generated_dir=str(generated_dir),
        scripts_dir=str(scripts_dir),
        model=str(model),
        readme=str(readme),
        requirements=str(requirements),
        generate_csv=str(generate_csv),
        output=str(root / "HW6_314511048.zip"),
        overwrite=False,
    )


def _write_png(path: Path) -> None:
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(path, format="PNG")


def _zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


if __name__ == "__main__":
    unittest.main()
