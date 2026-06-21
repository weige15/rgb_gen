from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.brainrot_data import ANIMALS, OBJECTS, BrainrotDataset, load_extra_records, load_train_records
from scripts.diffusion import DiffusionConfig, GaussianDiffusion
from scripts.model import ConditionalUNet, UNetConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scratch Brainrot conditional DDPM.")
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--image_dir", default="dataset/trainset")
    parser.add_argument("--extra_manifest")
    parser.add_argument("--output_model", default="model.pth")
    parser.add_argument("--run_dir", default="runs/train")
    parser.add_argument("--resume_from")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--devices")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--schedule", default="linear", choices=("linear", "cosine"))
    parser.add_argument("--train_timesteps", type=int, default=1000)
    parser.add_argument("--sampling_steps", type=int, default=1000)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--sample_every_steps", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channel_multipliers", default="1,2,4")
    parser.add_argument("--residual_blocks", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention_resolutions", default="")
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu_smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except Exception as exc:  # pragma: no cover - exercised through CLI tests.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        print(f"ERROR: {exc}")
        return 1
    return 0


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_training_args(args)
    device_ids = parse_device_ids(args.devices)
    distributed = _distributed_state()
    device = resolve_device(device_ids, args.cpu_smoke, distributed["local_rank"], distributed["world_size"])
    resume_path = Path(args.resume_from) if args.resume_from else None
    _prepare_outputs(Path(args.output_model), Path(args.run_dir), args.overwrite, resume_path)
    _set_seed(args.seed)

    if distributed["world_size"] > 1:
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    rank = distributed["rank"]
    resume_checkpoint = _load_training_checkpoint(resume_path, device) if resume_path is not None else None
    records = load_train_records(args.train_csv, args.image_dir)
    records.extend(load_extra_records(args.extra_manifest))
    dataset = BrainrotDataset(records, augment=True)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed["world_size"] > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    model_config, diffusion_config = _training_configs(args, resume_checkpoint)
    model = ConditionalUNet(model_config).to(device)
    diffusion = GaussianDiffusion(diffusion_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema_state = _clone_state_dict(model) if args.ema else None
    global_step, ema_state = _restore_training_state(resume_checkpoint, model, ema_state, optimizer, scaler)
    if global_step >= args.max_steps:
        raise ValueError("resume checkpoint train_step is already >= max_steps.")
    expected_steps = min(
        args.max_steps,
        global_step + args.epochs * (len(dataloader) // args.grad_accum_steps),
    )
    train_model = (
        DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" and device.index is not None else None)
        if distributed["world_size"] > 1
        else model
    )

    run_dir = Path(args.run_dir)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        event = _log_event(
            args,
            distributed,
            global_step,
            None,
            Path(args.output_model),
            "resume" if resume_path is not None else "start",
        )
        if resume_path is not None:
            event["resume_from"] = str(resume_path)
        _append_log(run_dir / "train.jsonl", event)

    last_loss: float | None = None
    start_step = global_step
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_index, (images, conditions) in enumerate(dataloader):
            images = images.to(device)
            conditions = {key: value.to(device) for key, value in conditions.items()}
            autocast = torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda")
            with autocast if args.amp and device.type == "cuda" else nullcontext():
                loss = diffusion.training_loss(train_model, images, conditions) / args.grad_accum_steps
            scaler.scale(loss).backward()

            if (batch_index + 1) % args.grad_accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            _ensure_finite_gradient(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = float(loss.detach().cpu().item() * args.grad_accum_steps)
            if ema_state is not None:
                _update_ema(ema_state, model, args.ema_decay)

            if rank == 0:
                event = _log_event(args, distributed, global_step, last_loss, Path(args.output_model), "step")
                event["epoch"] = epoch
                _add_progress_timing(event, started, global_step - start_step, expected_steps - start_step)
                _append_log(run_dir / "train.jsonl", event)
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    checkpoint_path = run_dir / "checkpoints" / f"step_{global_step:06d}.pth"
                    if checkpoint_path.exists() and not args.overwrite:
                        raise FileExistsError(f"Checkpoint {checkpoint_path} exists; pass --overwrite to replace it.")
                    _save_checkpoint(
                        checkpoint_path,
                        model,
                        ema_state,
                        optimizer,
                        scaler,
                        model_config,
                        diffusion_config,
                        args,
                        global_step,
                    )

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    if global_step == start_step:
        raise RuntimeError("Training finished without completing an optimizer step.")

    if rank == 0:
        checkpoint = _save_checkpoint(
            Path(args.output_model),
            model,
            ema_state,
            optimizer,
            scaler,
            model_config,
            diffusion_config,
            args,
            global_step,
        )
        event = _log_event(args, distributed, global_step, last_loss, Path(args.output_model), "save")
        _add_progress_timing(event, started, global_step - start_step, global_step - start_step)
        _append_log(run_dir / "train.jsonl", event)
    else:
        checkpoint = {}

    if distributed["world_size"] > 1:
        torch.distributed.destroy_process_group()
    return checkpoint


def parse_device_ids(devices: str | None) -> list[int]:
    if not devices:
        return []
    parsed = [int(part.strip()) for part in devices.split(",") if part.strip() != ""]
    if len(parsed) != len(set(parsed)):
        raise ValueError("devices must not contain duplicates.")
    if any(device_id < 0 for device_id in parsed):
        raise ValueError("devices must be non-negative CUDA indices.")
    if len(parsed) > 4:
        raise ValueError("at most 4 GPUs may be selected.")
    return parsed


def resolve_device(device_ids: list[int], cpu_smoke: bool, local_rank: int, world_size: int) -> torch.device:
    if world_size > 4:
        raise ValueError("torchrun world size must be <= 4.")
    if cpu_smoke:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; pass --cpu_smoke for a CPU smoke run.")
    if device_ids:
        if world_size > len(device_ids):
            raise ValueError("torchrun world size exceeds selected device count.")
        if max(device_ids) >= torch.cuda.device_count():
            raise ValueError("selected CUDA device index is unavailable.")
        device = torch.device(f"cuda:{device_ids[local_rank]}")
    else:
        device = torch.device(f"cuda:{local_rank}") if world_size > 1 else torch.device("cuda")
    if device.index is not None:
        torch.cuda.set_device(device)
    return device


def _validate_training_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive.")
    if not 0.0 <= args.ema_decay <= 1.0:
        raise ValueError("ema_decay must be in [0, 1].")
    if args.sample_every_steps:
        raise ValueError("sample_every_steps is reserved for a later sample-grid task.")


def _prepare_outputs(output_model: Path, run_dir: Path, overwrite: bool, resume_from: Path | None = None) -> None:
    if output_model.exists() and not overwrite:
        raise FileExistsError(f"Output model {output_model} exists; pass --overwrite to replace it.")
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite and resume_from is None:
        raise FileExistsError(f"Run directory {run_dir} is not empty; pass --overwrite to reuse it.")
    output_model.parent.mkdir(parents=True, exist_ok=True)


def _distributed_state() -> dict[str, int]:
    return {
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_int_tuple(value: str, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers.") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty.")
    return parsed


def _parse_optional_int_tuple(value: str, name: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return _parse_int_tuple(value, name)


def _ensure_finite_gradient(model: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise FloatingPointError("No gradients were produced during training.")
    if not any(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise FloatingPointError("No finite gradients were produced during training.")


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[key].copy_(value)


def _training_configs(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> tuple[UNetConfig, DiffusionConfig]:
    if checkpoint is not None:
        return (
            UNetConfig(**dict(checkpoint["model_config"])),
            DiffusionConfig(**dict(checkpoint["diffusion_config"])),
        )
    return (
        UNetConfig(
            base_channels=args.base_channels,
            channel_multipliers=_parse_int_tuple(args.channel_multipliers, "channel_multipliers"),
            residual_blocks=args.residual_blocks,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            attention_resolutions=_parse_optional_int_tuple(args.attention_resolutions, "attention_resolutions"),
            attention_heads=args.attention_heads,
        ),
        DiffusionConfig(
            train_timesteps=args.train_timesteps,
            schedule=args.schedule,
            sampling_steps=args.sampling_steps,
            cfg_dropout=args.cfg_dropout,
        ),
    )


def _load_training_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint {path} does not exist.")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint must be a dictionary.")
    required = {
        "format_version",
        "model_state_dict",
        "model_config",
        "diffusion_config",
        "animals",
        "objects",
        "train_step",
        "uses_pretrained_generator_weights",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Resume checkpoint is missing required keys: {missing}.")
    if checkpoint["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format version {checkpoint['format_version']!r}.")
    if list(checkpoint["animals"]) != list(ANIMALS) or list(checkpoint["objects"]) != list(OBJECTS):
        raise ValueError("Resume checkpoint category mapping does not match this project.")
    if checkpoint["uses_pretrained_generator_weights"]:
        raise ValueError("Refusing to resume from a pretrained-generator checkpoint.")
    return checkpoint


def _restore_training_state(
    checkpoint: dict[str, Any] | None,
    model: torch.nn.Module,
    ema_state: dict[str, torch.Tensor] | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, dict[str, torch.Tensor] | None]:
    if checkpoint is None:
        return 0, ema_state
    model.load_state_dict(checkpoint["model_state_dict"])
    if ema_state is not None:
        saved_ema = checkpoint.get("ema_state_dict")
        ema_state = (
            {key: value.detach().clone() for key, value in saved_ema.items()}
            if saved_ema is not None
            else _clone_state_dict(model)
        )
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint["train_step"]), ema_state


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema_state: dict[str, torch.Tensor] | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: UNetConfig,
    diffusion_config: DiffusionConfig,
    args: argparse.Namespace,
    train_step: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": asdict(model_config),
        "diffusion_config": asdict(diffusion_config),
        "animals": ANIMALS,
        "objects": OBJECTS,
        "seed": args.seed,
        "train_step": train_step,
        "uses_pretrained_generator_weights": False,
        "training_config": _jsonable_args(args),
    }
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)
    return checkpoint


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _log_event(
    args: argparse.Namespace,
    distributed: dict[str, int],
    step: int,
    loss: float | None,
    output_model: Path,
    event: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "seed": args.seed,
        "device": args.devices or ("cpu" if args.cpu_smoke else "auto"),
        "devices": args.devices or ("cpu" if args.cpu_smoke else "auto"),
        "rank": distributed["rank"],
        "world_size": distributed["world_size"],
        "step": step,
        "max_steps": args.max_steps,
        "loss": loss,
        "elapsed_seconds": None,
        "eta_seconds": None,
        "output_model": str(output_model),
    }


def _add_progress_timing(event: dict[str, Any], started: float, step: int, expected_steps: int) -> None:
    elapsed = time.monotonic() - started
    event["elapsed_seconds"] = round(elapsed, 4)
    if step > 0 and expected_steps >= step:
        event["eta_seconds"] = round((elapsed / step) * (expected_steps - step), 4)


def _append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
