from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.brainrot_data import ANIMALS, OBJECTS, GenerationRequest, condition_to_ids, load_generation_requests
from scripts.diffusion import DiffusionConfig, GaussianDiffusion, tensor_to_uint8_images
from scripts.model import ConditionalUNet, UNetConfig
from scripts.train import parse_device_ids, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Brainrot images from a scratch DDPM checkpoint.")
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--generate_csv", default="dataset/generate.csv")
    parser.add_argument("--output_dir", default="generated_images")
    parser.add_argument("--run_dir", default="runs/generate")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--devices")
    parser.add_argument("--sampler", default="ddpm")
    parser.add_argument("--sampling_steps", type=int)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu_smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        generate(parse_args(argv))
    except Exception as exc:  # pragma: no cover - exercised by CLI tests.
        print(f"ERROR: {exc}")
        return 1
    return 0


@torch.no_grad()
def generate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    distributed = _distributed_state()
    device_ids = parse_device_ids(args.devices)
    device = resolve_device(device_ids, args.cpu_smoke, distributed["local_rank"], distributed["world_size"])
    if distributed["world_size"] > 1:
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    try:
        output_dir = Path(args.output_dir)
        run_dir = Path(args.run_dir)
        _prepare_output_dir(output_dir, args.overwrite)
        if distributed["rank"] == 0:
            run_dir.mkdir(parents=True, exist_ok=True)

        requests = load_generation_requests(args.generate_csv)
        if args.limit is not None:
            requests = requests[: args.limit]
        rank_requests = partition_requests(requests, distributed["rank"], distributed["world_size"])

        checkpoint = _load_checkpoint(Path(args.model), device)
        model, diffusion = _build_from_checkpoint(checkpoint, args, device)
        model.eval()

        started = time.monotonic()
        generated = 0
        for batch in _batches(rank_requests, args.batch_size):
            images = _sample_requests(model, diffusion, batch, args.seed, args, device)
            for request, image in zip(batch, images, strict=True):
                image_path = output_dir / request.image_id
                if image_path.exists() and not args.overwrite:
                    raise FileExistsError(f"{image_path} exists; pass --overwrite to replace it.")
                image.save(image_path, format="PNG")
                generated += 1
                if distributed["rank"] == 0:
                    event = _log_event(args, distributed, generated, len(rank_requests), output_dir, started)
                    _append_log(run_dir / "generate.jsonl", event)
                    if not args.quiet:
                        print(f"generated={generated}/{len(rank_requests)} file={request.image_id}")

        if distributed["world_size"] > 1:
            torch.distributed.barrier()
        return {
            "generated": generated,
            "expected_for_rank": len(rank_requests),
            "output_dir": str(output_dir),
        }
    finally:
        if distributed["world_size"] > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def partition_requests(
    requests: Sequence[GenerationRequest],
    rank: int,
    world_size: int,
) -> list[GenerationRequest]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size).")
    return list(requests[rank::world_size])


def _batches(requests: Sequence[GenerationRequest], batch_size: int) -> list[list[GenerationRequest]]:
    return [list(requests[index : index + batch_size]) for index in range(0, len(requests), batch_size)]


def _sample_requests(
    model: ConditionalUNet,
    diffusion: GaussianDiffusion,
    requests: Sequence[GenerationRequest],
    base_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[Image.Image]:
    if not requests:
        return []
    condition_values = [condition_to_ids(request.condition) for request in requests]
    condition_ids = {
        key: torch.tensor([row[key] for row in condition_values], dtype=torch.long, device=device)
        for key in ("animal_id", "object_id", "pair_id")
    }
    generator = torch.Generator(device=device.type).manual_seed(base_seed + requests[0].row_index)
    sample = diffusion.sample(
        model,
        condition_ids,
        (len(requests), 3, 64, 64),
        sampler=args.sampler,
        steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        device=device,
    )
    return [
        Image.frombytes("RGB", (uint8.shape[1], uint8.shape[0]), uint8.numpy().tobytes())
        for uint8 in tensor_to_uint8_images(sample.cpu())
    ]


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist.")
    checkpoint = torch.load(path, map_location=device)
    required = {
        "format_version",
        "model_state_dict",
        "model_config",
        "diffusion_config",
        "animals",
        "objects",
        "uses_pretrained_generator_weights",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint {path} is missing required keys: {missing!r}.")
    if checkpoint["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format version {checkpoint['format_version']!r}.")
    if list(checkpoint["animals"]) != list(ANIMALS) or list(checkpoint["objects"]) != list(OBJECTS):
        raise ValueError("Checkpoint category mapping does not match assignment categories.")
    if checkpoint["uses_pretrained_generator_weights"]:
        raise ValueError("Checkpoint declares pretrained generator weights; refusing to generate.")
    return checkpoint


def _build_from_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalUNet, GaussianDiffusion]:
    model_config = dict(checkpoint["model_config"])
    if "channel_multipliers" in model_config:
        model_config["channel_multipliers"] = tuple(model_config["channel_multipliers"])
    model = ConditionalUNet(UNetConfig(**model_config)).to(device)
    state_key = "ema_state_dict" if args.use_ema else "model_state_dict"
    state_dict = checkpoint.get(state_key)
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain {state_key}.")
    model.load_state_dict(state_dict)

    diffusion_config = DiffusionConfig(**dict(checkpoint["diffusion_config"]))
    if args.sampling_steps is not None:
        diffusion_config = replace(diffusion_config, sampling_steps=args.sampling_steps)
    diffusion_config = replace(diffusion_config, sampler=args.sampler, guidance_scale=args.guidance_scale)
    return model, GaussianDiffusion(diffusion_config)


def _validate_args(args: argparse.Namespace) -> None:
    parse_device_ids(args.devices)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when provided.")
    if args.guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative.")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory {output_dir} is not empty; pass --overwrite to reuse it.")
    output_dir.mkdir(parents=True, exist_ok=True)


def _distributed_state() -> dict[str, int]:
    return {
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def _log_event(
    args: argparse.Namespace,
    distributed: dict[str, int],
    generated: int,
    expected: int,
    output_dir: Path,
    started: float,
) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    eta = None
    if generated > 0 and expected >= generated:
        eta = round((elapsed / generated) * (expected - generated), 4)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": args.devices or ("cpu" if args.cpu_smoke else "auto"),
        "rank": distributed["rank"],
        "world_size": distributed["world_size"],
        "generated": generated,
        "expected": expected,
        "elapsed_seconds": round(elapsed, 4),
        "eta_seconds": eta,
        "output_dir": str(output_dir),
    }


def _append_log(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
