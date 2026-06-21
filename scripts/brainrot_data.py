from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset


ANIMALS: tuple[str, ...] = (
    "shark",
    "crocodile",
    "frog",
    "cat",
    "dog",
    "capybara",
    "elephant",
    "bird",
    "fish",
    "monkey",
)

OBJECTS: tuple[str, ...] = (
    "sneaker",
    "airplane",
    "coffee cup",
    "banana",
    "cactus",
    "toilet",
    "pizza",
    "drum",
    "car",
    "chair",
)

ANIMAL_TO_ID = {label: index for index, label in enumerate(ANIMALS)}
OBJECT_TO_ID = {label: index for index, label in enumerate(OBJECTS)}
IMAGE_SIZE = 64
EXTRA_DATA_COLUMNS: tuple[str, ...] = (
    "image_path",
    "animal",
    "object",
    "source",
    "retrieval_date",
    "license_or_allowance",
    "label_author_or_method",
    "image_hash",
    "preprocessing_notes",
)


class CSVValidationError(ValueError):
    pass


class ImageValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    animal: str
    object: str
    animal_id: int
    object_id: int
    pair_id: int


@dataclass(frozen=True)
class TrainRecord:
    image_id: str
    image_path: Path
    condition: Condition
    source: str = "dataset"


@dataclass(frozen=True)
class GenerationRequest:
    image_id: str
    prompt: str
    condition: Condition
    row_index: int


def build_condition(animal: str, object_name: str) -> Condition:
    if animal not in ANIMAL_TO_ID:
        raise CSVValidationError(f"Unknown animal label {animal!r}.")
    if object_name not in OBJECT_TO_ID:
        raise CSVValidationError(f"Unknown object label {object_name!r}.")

    animal_id = ANIMAL_TO_ID[animal]
    object_id = OBJECT_TO_ID[object_name]
    return Condition(
        animal=animal,
        object=object_name,
        animal_id=animal_id,
        object_id=object_id,
        pair_id=animal_id * len(OBJECTS) + object_id,
    )


def expected_prompt(animal: str, object_name: str) -> str:
    return f"a {animal} and a {object_name}"


def validate_prompt(animal: str, object_name: str, prompt: str) -> None:
    expected = expected_prompt(animal, object_name)
    if prompt != expected:
        raise CSVValidationError(
            f"Prompt {prompt!r} does not match expected prompt {expected!r}."
        )


def condition_to_ids(condition: Condition) -> dict[str, int]:
    return {
        "animal_id": condition.animal_id,
        "object_id": condition.object_id,
        "pair_id": condition.pair_id,
    }


def load_train_records(
    train_csv: str | Path,
    image_dir: str | Path,
    image_size: int = IMAGE_SIZE,
) -> list[TrainRecord]:
    csv_path = Path(train_csv)
    image_root = Path(image_dir)
    rows = _read_strict_csv(csv_path, ("id", "animal", "object"))
    records: list[TrainRecord] = []
    seen_ids: set[str] = set()

    for row_index, row in enumerate(rows):
        context = f"{csv_path}:{row_index + 2}"
        _reject_extra_row_values(row, context)
        image_id = _required(row, "id", context)
        if image_id in seen_ids:
            raise CSVValidationError(f"{context}: duplicate id {image_id!r}.")
        seen_ids.add(image_id)
        _validate_png_filename(image_id, context)

        condition = _condition_from_row(row, context)
        image_path = image_root / image_id
        validate_image_file(image_path, image_size=image_size)
        records.append(TrainRecord(image_id=image_id, image_path=image_path, condition=condition))

    return records


def load_generation_requests(generate_csv: str | Path) -> list[GenerationRequest]:
    csv_path = Path(generate_csv)
    rows = _read_strict_csv(csv_path, ("id", "animal", "object", "prompt"))
    requests: list[GenerationRequest] = []
    seen_ids: set[str] = set()

    for row_index, row in enumerate(rows):
        context = f"{csv_path}:{row_index + 2}"
        _reject_extra_row_values(row, context)
        image_id = _required(row, "id", context)
        if image_id in seen_ids:
            raise CSVValidationError(f"{context}: duplicate id {image_id!r}.")
        seen_ids.add(image_id)
        _validate_png_filename(image_id, context)

        condition = _condition_from_row(row, context)
        prompt = _required(row, "prompt", context)
        try:
            validate_prompt(condition.animal, condition.object, prompt)
        except CSVValidationError as exc:
            raise CSVValidationError(f"{context}: {exc}") from exc
        requests.append(
            GenerationRequest(
                image_id=image_id,
                prompt=prompt,
                condition=condition,
                row_index=row_index,
            )
        )

    return requests


def load_extra_records(
    manifest_csv: str | Path | None,
    image_size: int = IMAGE_SIZE,
) -> list[TrainRecord]:
    if manifest_csv is None:
        return []

    csv_path = Path(manifest_csv)
    rows = _read_strict_csv(csv_path, EXTRA_DATA_COLUMNS)
    records: list[TrainRecord] = []
    seen_paths: set[Path] = set()

    for row_index, row in enumerate(rows):
        context = f"{csv_path}:{row_index + 2}"
        _reject_extra_row_values(row, context)
        image_path = _resolve_manifest_image_path(csv_path, _required(row, "image_path", context))
        if image_path in seen_paths:
            raise CSVValidationError(f"{context}: duplicate image path {str(image_path)!r}.")
        seen_paths.add(image_path)

        source = _required(row, "source", context)
        retrieval_date = _required(row, "retrieval_date", context)
        allowance = _required(row, "license_or_allowance", context)
        label_method = _required(row, "label_author_or_method", context)
        image_hash = _required(row, "image_hash", context)
        preprocessing_notes = _required(row, "preprocessing_notes", context)
        _reject_pretrained_generated_text(
            (source, retrieval_date, allowance, label_method, preprocessing_notes),
            context,
        )

        condition = _condition_from_row(row, context)
        validate_image_file(image_path, image_size=image_size)
        _validate_sha256(image_path, image_hash, context)
        records.append(
            TrainRecord(
                image_id=image_path.name,
                image_path=image_path,
                condition=condition,
                source=source,
            )
        )

    return records


def validate_extra_manifest(manifest_csv: str | Path | None) -> list[str]:
    load_extra_records(manifest_csv)
    return []


def validate_image_file(image_path: str | Path, image_size: int = IMAGE_SIZE) -> None:
    _open_validated_image(Path(image_path), image_size=image_size).close()


class BrainrotDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TrainRecord],
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
    ):
        self.records = list(records)
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, int]]:
        record = self.records[index]
        with _open_validated_image(record.image_path, image_size=self.image_size) as image:
            if self.augment:
                image = _augment_image(image)
            tensor = image_to_tensor(image, image_size=self.image_size)
        return tensor, condition_to_ids(record.condition)


def image_to_tensor(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    if image.mode != "RGB":
        raise ImageValidationError(f"Image mode {image.mode!r} is not RGB.")
    if image.size != (image_size, image_size):
        raise ImageValidationError(
            f"Image size {image.size} does not match expected {(image_size, image_size)}."
        )
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.reshape(image_size, image_size, 3).permute(2, 0, 1)
    return data.to(dtype=torch.float32).div(127.5).sub(1.0)


def _condition_from_row(row: dict[str, str], context: str) -> Condition:
    animal = _required(row, "animal", context)
    object_name = _required(row, "object", context)
    try:
        return build_condition(animal, object_name)
    except CSVValidationError as exc:
        raise CSVValidationError(f"{context}: {exc}") from exc


def _read_strict_csv(path: Path, expected_columns: Iterable[str]) -> list[dict[str, str]]:
    expected = list(expected_columns)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise CSVValidationError(f"{path}: CSV file is empty.")
            if reader.fieldnames != expected:
                raise CSVValidationError(
                    f"{path}: expected columns {expected!r}, found {reader.fieldnames!r}."
                )
            rows = list(reader)
    except FileNotFoundError:
        raise

    if not rows:
        raise CSVValidationError(f"{path}: CSV file has no data rows.")
    return rows


def _reject_extra_row_values(row: dict[str, str], context: str) -> None:
    if row.get(None):
        raise CSVValidationError(f"{context}: row has extra values {row[None]!r}.")


def _required(row: dict[str, str], key: str, context: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise CSVValidationError(f"{context}: missing required value {key!r}.")
    return value


def _validate_png_filename(image_id: str, context: str) -> None:
    path = Path(image_id)
    if path.name != image_id or path.suffix.lower() != ".png":
        raise CSVValidationError(f"{context}: id {image_id!r} must be a PNG filename.")


def _resolve_manifest_image_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_sha256(image_path: Path, expected_hash: str, context: str) -> None:
    normalized = expected_hash.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise CSVValidationError(f"{context}: image_hash must be a SHA-256 hex digest.")

    digest = hashlib.sha256()
    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != normalized:
        raise CSVValidationError(
            f"{context}: image_hash does not match {image_path}; expected {normalized}, got {actual}."
        )


def _reject_pretrained_generated_text(values: Iterable[str], context: str) -> None:
    text = " ".join(value.lower() for value in values)
    prohibited = (
        "pretrained",
        "pre-trained",
        "model-generated",
        "model generated",
        "ai-generated",
        "ai generated",
        "stable diffusion",
        "midjourney",
        "dall-e",
        "dalle",
        "openclip",
        "clip-labeled",
        "clip labeled",
    )
    for marker in prohibited:
        if marker in text:
            raise CSVValidationError(
                f"{context}: extra data provenance suggests pretrained/generated content: {marker!r}."
            )


def _open_validated_image(image_path: Path, image_size: int = IMAGE_SIZE) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    if not image_path.is_file():
        raise ImageValidationError(f"Image path {image_path} is not a file.")

    try:
        with Image.open(image_path) as image:
            image.load()
            if image.format != "PNG":
                raise ImageValidationError(
                    f"Image {image_path} has format {image.format!r}, expected 'PNG'."
                )
            if image.mode != "RGB":
                raise ImageValidationError(
                    f"Image {image_path} has mode {image.mode!r}, expected 'RGB'."
                )
            if image.size != (image_size, image_size):
                raise ImageValidationError(
                    f"Image {image_path} has size {image.size}, expected "
                    f"{image_size}x{image_size}."
                )
            return image.copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageValidationError(f"Image {image_path} could not be read as an image.") from exc


def _augment_image(image: Image.Image) -> Image.Image:
    augmented = image
    if random.random() < 0.5:
        augmented = ImageOps.mirror(augmented)
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-0.05, 0.05)
        augmented = enhancer_cls(augmented).enhance(factor)
    return augmented
