from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable

from scripts.validate_outputs import validate_outputs


DEFAULT_STUDENT_ID = "314511048"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the HW6 Brainrot E3 submission archive.")
    parser.add_argument("--student_id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--generated_dir", default="generated_images")
    parser.add_argument("--scripts_dir", default="scripts")
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--generate_csv", default="dataset/generate.csv")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        package_submission(parse_args(argv))
    except Exception as exc:  # pragma: no cover - CLI behavior.
        print(f"ERROR: {exc}")
        return 1
    return 0


def package_submission(args: argparse.Namespace, expected_count: int = 2000) -> dict[str, int | str]:
    output = Path(args.output or f"HW6_{args.student_id}.zip")
    expected_name = f"HW6_{args.student_id}.zip"
    if output.name != expected_name:
        raise ValueError(f"Output archive must be named {expected_name}.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Archive {output} exists; pass --overwrite to replace it.")

    generated_dir = _require_dir(args.generated_dir, "generated images")
    scripts_dir = _require_dir(args.scripts_dir, "scripts")
    model = _require_file(args.model, "model")
    readme = _require_file(args.readme, "README")
    requirements = _require_file(args.requirements, "requirements")

    errors = validate_outputs(
        generate_csv=args.generate_csv,
        image_dir=generated_dir,
        expected_count=expected_count,
        strict_prompt=True,
    )
    if errors:
        raise ValueError("Generated outputs failed validation: " + "; ".join(errors))

    output.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(generated_dir.glob("*.png")):
            archive.write(file_path, Path("generated_images") / file_path.name)
            file_count += 1
        for file_path in _iter_packaged_scripts(scripts_dir):
            archive.write(file_path, Path("scripts") / file_path.relative_to(scripts_dir))
            file_count += 1
        archive.write(model, "model.pth")
        archive.write(readme, "README.md")
        archive.write(requirements, "requirements.txt")
        file_count += 3
    return {"archive": str(output), "file_count": file_count}


def _iter_packaged_scripts(scripts_dir: Path) -> Iterable[Path]:
    excluded_parts = {"__pycache__", "checkpoints", "runs"}
    for path in sorted(scripts_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(scripts_dir)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name.startswith("scores") and path.suffix == ".json":
            continue
        yield path


def _require_file(path: str | Path, label: str) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing required {label} file: {file_path}.")
    return file_path


def _require_dir(path: str | Path, label: str) -> Path:
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Missing required {label} directory: {dir_path}.")
    return dir_path


if __name__ == "__main__":
    raise SystemExit(main())
