from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.brainrot_data import load_generation_requests
from scripts.validate_outputs import validate_outputs


DEFAULT_SCORES = ("fid", "clip_t")
EVALUATOR_MODELS = {
    "FID": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
    "CLIP_T": "open_clip ViT-B-32-quickgelu/openai",
}
NOTES = "Local development scores only; Codabench remains authoritative."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score 2,000 generated Brainrot images for local development.")
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--generate_csv", default="dataset/generate.csv")
    parser.add_argument("--ref_mu", default="scoring_program/input/ref/test_mu.npy")
    parser.add_argument("--ref_sigma", default="scoring_program/input/ref/test_sigma.npy")
    parser.add_argument("--scores", nargs="+", default=list(DEFAULT_SCORES), choices=("fid", "clip_t"))
    parser.add_argument("--output", default="scores_2000.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        score(parse_args(argv))
    except Exception as exc:  # pragma: no cover - exercised through CLI tests.
        print(f"ERROR: {exc}")
        return 1
    return 0


def score(args: argparse.Namespace, expected_count: int = 2000) -> dict[str, Any]:
    scores = tuple(dict.fromkeys(score_name.lower() for score_name in args.scores))
    output_path = Path(args.output)
    requests = preflight_score_inputs(args, set(scores), output_path)
    _validate_submission_set(Path(args.image_dir), Path(args.generate_csv), expected_count=expected_count)
    requests = load_generation_requests(args.generate_csv)

    result: dict[str, Any] = {
        "num_images": expected_count,
        "image_dir": str(args.image_dir),
        "generate_csv": str(args.generate_csv),
        "scores_requested": list(scores),
        "evaluator_models": EVALUATOR_MODELS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": NOTES,
    }
    if "fid" in scores:
        result["FID"] = calculate_fid(
            image_dir=Path(args.image_dir),
            ref_mu=Path(args.ref_mu),
            ref_sigma=Path(args.ref_sigma),
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    if "clip_t" in scores:
        result["CLIP_T"] = calculate_clip_t(
            image_dir=Path(args.image_dir),
            requests=requests,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    write_report(result, output_path)
    return result


def preflight_score_inputs(
    args: argparse.Namespace,
    scores: set[str],
    output_path: Path,
) -> list[Any]:
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Score output {output_path} exists; pass --overwrite to replace it.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    requests: list[Any] = []
    if "clip_t" in scores:
        requests = load_generation_requests(args.generate_csv)
    if "fid" in scores:
        _require_file(args.ref_mu, "FID reference mean")
        _require_file(args.ref_sigma, "FID reference covariance")
    return requests


def _validate_submission_set(
    image_dir: Path,
    generate_csv: Path,
    expected_count: int = 2000,
) -> None:
    validation_errors = validate_outputs(
        generate_csv=generate_csv,
        image_dir=image_dir,
        expected_count=expected_count,
        strict_prompt=True,
    )
    if validation_errors:
        raise ValueError("Generated outputs failed validation: " + "; ".join(validation_errors))


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calculate_fid(
    image_dir: Path,
    ref_mu: Path,
    ref_sigma: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> float:
    import numpy as np
    import torch
    from PIL import Image
    from scipy import linalg
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import Inception_V3_Weights, inception_v3

    class FIDDataset(Dataset):
        def __init__(self, paths: list[Path]):
            self.paths = paths
            self.transform = transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> torch.Tensor:
            with Image.open(self.paths[index]) as image:
                return self.transform(image.convert("RGB"))

    torch_device = _torch_device(device)
    paths = sorted(image_dir.glob("*.png"))
    loader = DataLoader(FIDDataset(paths), batch_size=batch_size, num_workers=num_workers)
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = nn.Identity()
    model.eval().to(torch_device)
    features = []
    with torch.no_grad():
        for batch in loader:
            features.append(model(batch.to(torch_device)).cpu())
    generated_features = torch.cat(features, dim=0).numpy()
    fake_mu = np.mean(generated_features, axis=0)
    fake_sigma = np.cov(generated_features, rowvar=False)
    real_mu = np.load(ref_mu)
    real_sigma = np.load(ref_sigma)
    covmean, _ = linalg.sqrtm(real_sigma @ fake_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = real_mu - fake_mu
    return float(diff.dot(diff) + np.trace(real_sigma + fake_sigma - 2 * covmean))


def calculate_clip_t(
    image_dir: Path,
    requests: list[Any],
    device: str,
    batch_size: int,
    num_workers: int,
) -> float:
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class CLIPTextDataset(Dataset):
        def __init__(self, image_root: Path, request_items: list[Any], preprocess, tokenizer):
            self.image_root = image_root
            self.requests = request_items
            self.preprocess = preprocess
            self.tokenizer = tokenizer

        def __len__(self) -> int:
            return len(self.requests)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            request = self.requests[index]
            with Image.open(self.image_root / request.image_id) as image:
                image_tensor = self.preprocess(image.convert("RGB"))
            return image_tensor, self.tokenizer(request.prompt)[0]

    torch_device = _torch_device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu",
        pretrained="openai",
        device=torch_device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model.eval()
    loader = DataLoader(
        CLIPTextDataset(image_dir, requests, preprocess, tokenizer),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    scores = []
    with torch.no_grad():
        for images, tokens in loader:
            image_features = model.encode_image(images.to(torch_device))
            text_features = model.encode_text(tokens.to(torch_device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores.append((image_features * text_features).sum(dim=-1).cpu())
    return float(torch.cat(scores).mean().item())


def _require_file(path: str | Path, label: str) -> None:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} file {file_path} does not exist.")


def _torch_device(device: str):
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for scoring but is unavailable.")
    return torch_device


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
