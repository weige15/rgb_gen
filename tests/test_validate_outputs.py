from __future__ import annotations

import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

from scripts.validate_outputs import main, validate_outputs


class ValidateOutputsTests(unittest.TestCase):
    def test_valid_outputs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)

            errors = validate_outputs(generate_csv, image_dir, expected_count=2, strict_prompt=True)

        self.assertEqual(errors, [])

    def test_expected_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)

            errors = validate_outputs(generate_csv, image_dir, expected_count=3)

        self.assertTrue(any("Expected 3 filenames" in error for error in errors))

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            (image_dir / "000002.png").unlink()

            errors = validate_outputs(generate_csv, image_dir, expected_count=2)

        self.assertTrue(any("Missing expected PNG files" in error for error in errors))

    def test_extra_png_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            _write_png(image_dir / "999999.png")

            errors = validate_outputs(generate_csv, image_dir, expected_count=2)

        self.assertTrue(any("Unexpected PNG files" in error for error in errors))

    def test_wrong_extension_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            _write_png(image_dir / "000003.jpg")

            errors = validate_outputs(generate_csv, image_dir, expected_count=2)

        self.assertTrue(any("Non-PNG files" in error for error in errors))

    def test_invalid_png_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            (image_dir / "000001.png").write_text("not an image", encoding="utf-8")

            errors = validate_outputs(generate_csv, image_dir, expected_count=2)

        self.assertTrue(any("could not be read" in error for error in errors))

    def test_wrong_dimensions_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            _write_png(image_dir / "000001.png", size=(63, 64))

            errors = validate_outputs(generate_csv, image_dir, expected_count=2)

        self.assertTrue(any("size is" in error for error in errors))

    def test_wrong_color_mode_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            Image.new("L", (64, 64), color=128).save(image_dir / "000001.png", format="PNG")

            errors = validate_outputs(generate_csv, image_dir, expected_count=2)

        self.assertTrue(any("mode is" in error for error in errors))

    def test_strict_prompt_rejects_bad_prompt_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            _write_generate_csv(
                generate_csv,
                [["000001.png", "fish", "chair", "fish chair"], ["000002.png", "cat", "toilet", "a cat and a toilet"]],
            )

            errors = validate_outputs(generate_csv, image_dir, expected_count=2, strict_prompt=True)

        self.assertTrue(any("Prompt" in error for error in errors))

    def test_cli_returns_zero_for_valid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--generate_csv",
                        str(generate_csv),
                        "--image_dir",
                        str(image_dir),
                        "--expected_count",
                        "2",
                        "--strict_prompt",
                    ]
                )

        self.assertEqual(exit_code, 0)

    def test_cli_returns_nonzero_for_invalid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_csv, image_dir = _good_fixture(root)
            (image_dir / "000002.png").unlink()

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--generate_csv",
                        str(generate_csv),
                        "--image_dir",
                        str(image_dir),
                        "--expected_count",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 1)


def _good_fixture(root: Path) -> tuple[Path, Path]:
    generate_csv = root / "generate.csv"
    image_dir = root / "images"
    image_dir.mkdir()
    _write_generate_csv(
        generate_csv,
        [
            ["000001.png", "fish", "chair", "a fish and a chair"],
            ["000002.png", "cat", "toilet", "a cat and a toilet"],
        ],
    )
    _write_png(image_dir / "000001.png")
    _write_png(image_dir / "000002.png")
    return generate_csv, image_dir


def _write_generate_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "animal", "object", "prompt"])
        writer.writerows(rows)


def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, color=(42, 120, 220)).save(path, format="PNG")


if __name__ == "__main__":
    unittest.main()
