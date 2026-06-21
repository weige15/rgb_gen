from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from scripts.brainrot_data import ANIMALS, OBJECTS


@dataclass(frozen=True)
class UNetConfig:
    image_channels: int = 3
    image_size: int = 64
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    residual_blocks: int = 1
    embedding_dim: int = 128
    dropout: float = 0.0
    attention_resolutions: tuple[int, ...] = ()
    attention_heads: int = 4
    num_animals: int = len(ANIMALS)
    num_objects: int = len(OBJECTS)
    num_pairs: int = len(ANIMALS) * len(OBJECTS)
    cfg_null_token: bool = True


class ConditionalUNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        _validate_config(config)
        self.config = config
        self.register_buffer(
            "_time_frequencies",
            _timestep_frequencies(config.embedding_dim),
            persistent=False,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        animal_count = config.num_animals + int(config.cfg_null_token)
        object_count = config.num_objects + int(config.cfg_null_token)
        pair_count = config.num_pairs + int(config.cfg_null_token)
        self.animal_embedding = nn.Embedding(animal_count, config.embedding_dim)
        self.object_embedding = nn.Embedding(object_count, config.embedding_dim)
        self.pair_embedding = nn.Embedding(pair_count, config.embedding_dim)
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        channels = [config.base_channels * multiplier for multiplier in config.channel_multipliers]
        resolutions = [config.image_size // (2**level) for level in range(len(channels))]
        self.input_conv = nn.Conv2d(config.image_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        current_channels = channels[0]
        for level, (out_channels, resolution) in enumerate(zip(channels, resolutions, strict=True)):
            blocks = []
            for block_index in range(config.residual_blocks):
                block_in = current_channels if block_index == 0 else out_channels
                blocks.append(ResidualBlock(block_in, out_channels, config.embedding_dim, config.dropout))
                if resolution in config.attention_resolutions:
                    blocks.append(AttentionBlock(out_channels, config.attention_heads))
            downsample = (
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
                if level < len(channels) - 1
                else nn.Identity()
            )
            self.down_blocks.append(nn.ModuleDict({"blocks": nn.ModuleList(blocks), "downsample": downsample}))
            current_channels = out_channels

        middle_blocks: list[nn.Module] = [
            ResidualBlock(current_channels, current_channels, config.embedding_dim, config.dropout)
        ]
        if resolutions[-1] in config.attention_resolutions:
            middle_blocks.append(AttentionBlock(current_channels, config.attention_heads))
        middle_blocks.append(ResidualBlock(current_channels, current_channels, config.embedding_dim, config.dropout))
        self.middle = nn.ModuleList(middle_blocks)

        self.up_blocks = nn.ModuleList()
        for skip_channels, resolution in zip(reversed(channels[:-1]), reversed(resolutions[:-1]), strict=True):
            upsample = nn.ConvTranspose2d(current_channels, skip_channels, kernel_size=4, stride=2, padding=1)
            blocks = [
                ResidualBlock(skip_channels * 2, skip_channels, config.embedding_dim, config.dropout)
            ]
            if resolution in config.attention_resolutions:
                blocks.append(AttentionBlock(skip_channels, config.attention_heads))
            for _ in range(config.residual_blocks - 1):
                blocks.append(ResidualBlock(skip_channels, skip_channels, config.embedding_dim, config.dropout))
                if resolution in config.attention_resolutions:
                    blocks.append(AttentionBlock(skip_channels, config.attention_heads))
            self.up_blocks.append(nn.ModuleDict({"upsample": upsample, "blocks": nn.ModuleList(blocks)}))
            current_channels = skip_channels

        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(current_channels), current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, config.image_channels, kernel_size=3, padding=1),
        )

    @property
    def null_animal_id(self) -> int:
        return self.config.num_animals

    @property
    def null_object_id(self) -> int:
        return self.config.num_objects

    @property
    def null_pair_id(self) -> int:
        return self.config.num_pairs

    def null_condition_ids(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if not self.config.cfg_null_token:
            raise ValueError("CFG null tokens are disabled in this model config.")
        return {
            "animal_id": torch.full((batch_size,), self.null_animal_id, dtype=torch.long, device=device),
            "object_id": torch.full((batch_size,), self.null_object_id, dtype=torch.long, device=device),
            "pair_id": torch.full((batch_size,), self.null_pair_id, dtype=torch.long, device=device),
        }

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        pair_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = _validate_image_tensor(x_t, self.config)
        _validate_id_tensor("timesteps", timesteps, batch_size, lower=0, upper=None)
        animal_upper = self.config.num_animals + int(self.config.cfg_null_token)
        object_upper = self.config.num_objects + int(self.config.cfg_null_token)
        pair_upper = self.config.num_pairs + int(self.config.cfg_null_token)
        _validate_id_tensor("animal_ids", animal_ids, batch_size, lower=0, upper=animal_upper)
        _validate_id_tensor("object_ids", object_ids, batch_size, lower=0, upper=object_upper)
        _validate_id_tensor("pair_ids", pair_ids, batch_size, lower=0, upper=pair_upper)

        timestep_embedding = _sinusoidal_timestep_embedding(
            timesteps,
            self.config.embedding_dim,
            self._time_frequencies,
        )
        embedding = self.time_mlp(timestep_embedding)
        embedding = embedding + self.animal_embedding(animal_ids)
        embedding = embedding + self.object_embedding(object_ids)
        embedding = embedding + self.pair_embedding(pair_ids)
        embedding = self.condition_mlp(embedding)

        hidden = self.input_conv(x_t)
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            for block in down["blocks"]:
                hidden = block(hidden, embedding)
            skips.append(hidden)
            hidden = down["downsample"](hidden)

        for block in self.middle:
            hidden = block(hidden, embedding)

        usable_skips = list(reversed(skips[:-1]))
        for up, skip in zip(self.up_blocks, usable_skips, strict=True):
            hidden = up["upsample"](hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"Upsampled shape {tuple(hidden.shape[-2:])} does not match skip shape "
                    f"{tuple(skip.shape[-2:])}."
                )
            hidden = torch.cat([hidden, skip], dim=1)
            for block in up["blocks"]:
                hidden = block(hidden, embedding)

        return self.output(hidden)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.embedding_projection = nn.Linear(embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        hidden = hidden + self.embedding_projection(embedding)[:, :, None, None]
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = _attention_heads(channels, heads)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.norm(x).reshape(batch, channels, height * width)
        query, key, value = self.qkv(hidden).chunk(3, dim=1)
        head_dim = channels // self.heads
        query = query.reshape(batch, self.heads, head_dim, height * width)
        key = key.reshape(batch, self.heads, head_dim, height * width)
        value = value.reshape(batch, self.heads, head_dim, height * width)
        attention = torch.softmax(
            torch.einsum("bhcn,bhcm->bhnm", query * (head_dim**-0.5), key),
            dim=-1,
        )
        hidden = torch.einsum("bhnm,bhcm->bhcn", attention, value).reshape(batch, channels, height * width)
        return x + self.proj(hidden).reshape(batch, channels, height, width)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    return _sinusoidal_timestep_embedding(
        timesteps,
        embedding_dim,
        _timestep_frequencies(embedding_dim, device=timesteps.device),
    )


def _timestep_frequencies(embedding_dim: int, device: torch.device | None = None) -> torch.Tensor:
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return torch.empty((0,), device=device, dtype=torch.float32)
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32)
    exponent = exponent / max(half_dim - 1, 1)
    return torch.exp(exponent)


def _sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [batch], got {tuple(timesteps.shape)}.")
    args = timesteps.to(device=frequencies.device, dtype=torch.float32)[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def _validate_config(config: UNetConfig) -> None:
    if config.image_channels <= 0:
        raise ValueError("image_channels must be positive.")
    if config.image_size <= 0 or config.image_size % (2 ** (len(config.channel_multipliers) - 1)) != 0:
        raise ValueError("image_size must be positive and divisible by the U-Net downsampling factor.")
    if config.base_channels <= 0:
        raise ValueError("base_channels must be positive.")
    if not config.channel_multipliers or any(multiplier <= 0 for multiplier in config.channel_multipliers):
        raise ValueError("channel_multipliers must be non-empty positive integers.")
    if config.residual_blocks <= 0:
        raise ValueError("residual_blocks must be positive.")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if any(resolution <= 0 for resolution in config.attention_resolutions):
        raise ValueError("attention_resolutions must contain positive integers.")
    valid_resolutions = {config.image_size // (2**level) for level in range(len(config.channel_multipliers))}
    invalid_resolutions = sorted(set(config.attention_resolutions) - valid_resolutions)
    if invalid_resolutions:
        raise ValueError(
            f"attention_resolutions must be among {sorted(valid_resolutions)}, got {invalid_resolutions}."
        )
    if config.attention_heads <= 0:
        raise ValueError("attention_heads must be positive.")
    if config.num_animals <= 0 or config.num_objects <= 0 or config.num_pairs <= 0:
        raise ValueError("label counts must be positive.")


def _validate_image_tensor(x_t: torch.Tensor, config: UNetConfig) -> int:
    if x_t.ndim != 4:
        raise ValueError(f"x_t must have shape [batch, channels, height, width], got {tuple(x_t.shape)}.")
    batch_size, channels, height, width = x_t.shape
    expected = (config.image_channels, config.image_size, config.image_size)
    if (channels, height, width) != expected:
        raise ValueError(
            f"x_t must have per-sample shape {expected}, got {(channels, height, width)}."
        )
    return batch_size


def _validate_id_tensor(
    name: str,
    values: torch.Tensor,
    batch_size: int,
    lower: int,
    upper: int | None,
) -> None:
    if values.ndim != 1 or values.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [{batch_size}], got {tuple(values.shape)}.")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if values.dtype not in integer_dtypes:
        raise ValueError(f"{name} must be an integer tensor.")
    if values.numel() == 0:
        return
    minimum = int(values.min().item())
    maximum = int(values.max().item())
    if minimum < lower:
        raise ValueError(f"{name} contains id {minimum}, below allowed minimum {lower}.")
    if upper is not None and maximum >= upper:
        raise ValueError(f"{name} contains id {maximum}, expected ids below {upper}.")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _attention_heads(channels: int, requested_heads: int) -> int:
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1
