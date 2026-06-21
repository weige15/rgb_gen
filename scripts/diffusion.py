from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    image_size: int = 64
    train_timesteps: int = 1000
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_target: str = "epsilon"
    sampler: str = "ddpm"
    sampling_steps: int = 1000
    ddim_eta: float = 0.0
    cfg_dropout: float = 0.0
    guidance_scale: float = 1.0


@dataclass(frozen=True)
class ScheduleTensors:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_previous: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig):
        _validate_config(config)
        self.config = config
        self.schedule = build_schedule(config)

    def q_sample(self, x_0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if x_0.shape != noise.shape:
            raise ValueError(f"x_0 shape {tuple(x_0.shape)} must match noise shape {tuple(noise.shape)}.")
        sqrt_alpha_bars = _extract(self.schedule.sqrt_alpha_bars, timesteps, x_0.shape)
        sqrt_one_minus = _extract(self.schedule.sqrt_one_minus_alpha_bars, timesteps, x_0.shape)
        return sqrt_alpha_bars * x_0 + sqrt_one_minus * noise

    def training_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        conditions: Mapping[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.config.prediction_target != "epsilon":
            raise ValueError(f"Unsupported prediction target {self.config.prediction_target!r}.")
        if x_0.ndim != 4:
            raise ValueError(f"x_0 must have shape [batch, channels, height, width], got {tuple(x_0.shape)}.")

        batch_size = x_0.shape[0]
        timesteps = torch.randint(
            0,
            self.config.train_timesteps,
            (batch_size,),
            device=x_0.device,
            generator=generator,
        )
        noise = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype, generator=generator)
        x_t = self.q_sample(x_0, timesteps, noise)
        model_conditions = self._maybe_drop_conditions(model, conditions, batch_size, x_0.device, generator)
        prediction = model(
            x_t,
            timesteps,
            model_conditions["animal_id"],
            model_conditions["object_id"],
            model_conditions["pair_id"],
        )
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("Diffusion training loss is not finite.")
        return loss

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        conditions: Mapping[str, torch.Tensor],
        shape: tuple[int, int, int, int],
        sampler: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        sampler_name = sampler or self.config.sampler
        if sampler_name not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampler {sampler_name!r}; expected 'ddpm' or 'ddim'.")
        sampling_steps = steps or self.config.sampling_steps
        _validate_sampling_steps(sampling_steps, self.config.train_timesteps)
        scale = self.config.guidance_scale if guidance_scale is None else guidance_scale
        if len(shape) != 4:
            raise ValueError(f"sample shape must be [batch, channels, height, width], got {shape!r}.")

        sample_device = torch.device(device) if device is not None else next(model.parameters()).device
        x_t = torch.randn(shape, device=sample_device, generator=generator)
        model_conditions = _conditions_to_device(conditions, sample_device)
        timesteps = torch.linspace(
            self.config.train_timesteps - 1,
            0,
            sampling_steps,
            dtype=torch.long,
            device=sample_device,
        )

        for index, timestep in enumerate(timesteps):
            timestep_batch = torch.full((shape[0],), int(timestep.item()), dtype=torch.long, device=sample_device)
            predicted_noise = self._predict_noise(model, x_t, timestep_batch, model_conditions, scale)
            alpha_bars_t = _extract(self.schedule.alpha_bars, timestep_batch, x_t.shape)
            sqrt_one_minus = _extract(self.schedule.sqrt_one_minus_alpha_bars, timestep_batch, x_t.shape)
            predicted_x_0 = ((x_t - sqrt_one_minus * predicted_noise) / torch.sqrt(alpha_bars_t)).clamp(-1.0, 1.0)

            if sampler_name == "ddim":
                if index == len(timesteps) - 1:
                    x_t = predicted_x_0
                    continue
                previous_timestep = torch.full(
                    (shape[0],),
                    int(timesteps[index + 1].item()),
                    dtype=torch.long,
                    device=sample_device,
                )
                alpha_bars_previous_t = _extract(self.schedule.alpha_bars, previous_timestep, x_t.shape)
                sigma = self.config.ddim_eta * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_bars_previous_t)
                        / (1.0 - alpha_bars_t)
                        * (1.0 - alpha_bars_t / alpha_bars_previous_t),
                        min=0.0,
                    )
                )
                direction = torch.sqrt(torch.clamp(1.0 - alpha_bars_previous_t - sigma.square(), min=0.0))
                noise = torch.randn(shape, device=sample_device, generator=generator) if self.config.ddim_eta > 0 else 0.0
                x_t = torch.sqrt(alpha_bars_previous_t) * predicted_x_0 + direction * predicted_noise + sigma * noise
                continue

            betas_t = _extract(self.schedule.betas, timestep_batch, x_t.shape)
            alphas_t = _extract(self.schedule.alphas, timestep_batch, x_t.shape)
            alpha_bars_previous_t = _extract(self.schedule.alpha_bars_previous, timestep_batch, x_t.shape)
            model_mean = (
                betas_t * torch.sqrt(alpha_bars_previous_t) / (1.0 - alpha_bars_t) * predicted_x_0
                + torch.sqrt(alphas_t) * (1.0 - alpha_bars_previous_t) / (1.0 - alpha_bars_t) * x_t
            )

            posterior_variance = _extract(self.schedule.posterior_variance, timestep_batch, x_t.shape)
            noise = torch.randn(shape, device=sample_device, generator=generator)
            nonzero_mask = (timestep_batch != 0).to(dtype=x_t.dtype).reshape(shape[0], 1, 1, 1)
            x_t = model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

        return x_t

    def _maybe_drop_conditions(
        self,
        model: torch.nn.Module,
        conditions: Mapping[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        model_conditions = _conditions_to_device(conditions, device)
        if self.config.cfg_dropout <= 0:
            return model_conditions
        condition_model = _condition_model(model)
        if not hasattr(condition_model, "null_condition_ids"):
            raise ValueError("cfg_dropout requires a model with null_condition_ids().")

        null_conditions = condition_model.null_condition_ids(batch_size, device)
        drop_mask = torch.rand((batch_size,), device=device, generator=generator) < self.config.cfg_dropout
        return {
            key: torch.where(drop_mask, null_conditions[key], value)
            for key, value in model_conditions.items()
        }

    def _predict_noise(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: Mapping[str, torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return model(
                x_t,
                timesteps,
                conditions["animal_id"],
                conditions["object_id"],
                conditions["pair_id"],
            )
        condition_model = _condition_model(model)
        if not hasattr(condition_model, "null_condition_ids"):
            raise ValueError("Classifier-free guidance requires a model with null_condition_ids().")

        null_conditions = condition_model.null_condition_ids(x_t.shape[0], x_t.device)
        uncond = model(
            x_t,
            timesteps,
            null_conditions["animal_id"],
            null_conditions["object_id"],
            null_conditions["pair_id"],
        )
        cond = model(
            x_t,
            timesteps,
            conditions["animal_id"],
            conditions["object_id"],
            conditions["pair_id"],
        )
        return uncond + guidance_scale * (cond - uncond)


def build_schedule(config: DiffusionConfig) -> ScheduleTensors:
    if config.schedule == "linear":
        betas = torch.linspace(config.beta_start, config.beta_end, config.train_timesteps)
    elif config.schedule == "cosine":
        betas = _cosine_betas(config.train_timesteps)
    else:
        raise ValueError(f"Unsupported diffusion schedule {config.schedule!r}.")

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_previous = torch.cat([torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
    posterior_variance = posterior_variance.clamp(min=1e-20)
    schedule = ScheduleTensors(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_previous=alpha_bars_previous,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        posterior_variance=posterior_variance,
    )
    _validate_schedule(schedule)
    return schedule


def tensor_to_uint8_images(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 4:
        raise ValueError(f"samples must have shape [batch, channels, height, width], got {tuple(samples.shape)}.")
    if samples.shape[1] != 3:
        raise ValueError(f"samples must have 3 channels, got {samples.shape[1]}.")
    images = samples.detach().clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
    return images.to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()


def _condition_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def _cosine_betas(train_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = train_timesteps + 1
    x = torch.linspace(0, train_timesteps, steps)
    alpha_bars = torch.cos(((x / train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    return betas.clamp(min=1e-8, max=0.999)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [batch], got {tuple(timesteps.shape)}.")
    selected = values.to(device=timesteps.device, dtype=torch.float32)[timesteps]
    return selected.reshape(timesteps.shape[0], *((1,) * (len(broadcast_shape) - 1)))


def _conditions_to_device(
    conditions: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    required = ("animal_id", "object_id", "pair_id")
    missing = [key for key in required if key not in conditions]
    if missing:
        raise ValueError(f"conditions missing required keys: {missing!r}.")
    return {key: conditions[key].to(device=device, dtype=torch.long) for key in required}


def _validate_config(config: DiffusionConfig) -> None:
    if config.image_size <= 0:
        raise ValueError("image_size must be positive.")
    if config.train_timesteps <= 0:
        raise ValueError("train_timesteps must be positive.")
    if config.beta_start <= 0 or config.beta_end <= 0 or config.beta_start >= 1 or config.beta_end >= 1:
        raise ValueError("linear beta values must be in (0, 1).")
    if config.beta_start >= config.beta_end:
        raise ValueError("beta_start must be smaller than beta_end.")
    if config.prediction_target != "epsilon":
        raise ValueError("Only epsilon prediction is supported.")
    if config.sampler not in {"ddpm", "ddim"}:
        raise ValueError("sampler must be 'ddpm' or 'ddim'.")
    _validate_sampling_steps(config.sampling_steps, config.train_timesteps)
    if config.ddim_eta < 0:
        raise ValueError("ddim_eta must be non-negative.")
    if not 0.0 <= config.cfg_dropout <= 1.0:
        raise ValueError("cfg_dropout must be in [0, 1].")
    if config.guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative.")


def _validate_schedule(schedule: ScheduleTensors) -> None:
    for name, tensor in schedule.__dict__.items():
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Schedule tensor {name} contains non-finite values.")
    if not ((schedule.betas > 0).all() and (schedule.betas < 1).all()):
        raise ValueError("Schedule betas must be in (0, 1).")
    if not torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]).item():
        raise ValueError("Cumulative alphas must decrease monotonically.")


def _validate_sampling_steps(sampling_steps: int, train_timesteps: int) -> None:
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive.")
    if sampling_steps > train_timesteps:
        raise ValueError("sampling_steps must be <= train_timesteps.")
