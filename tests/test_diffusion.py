from __future__ import annotations

import unittest

import torch

from scripts.diffusion import DiffusionConfig, GaussianDiffusion, tensor_to_uint8_images
from scripts.model import ConditionalUNet, UNetConfig


class GaussianDiffusionTests(unittest.TestCase):
    def test_linear_schedule_is_finite_and_monotonic(self) -> None:
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=8, sampling_steps=4))
        schedule = diffusion.schedule

        self.assertTrue(torch.isfinite(schedule.betas).all().item())
        self.assertTrue(((schedule.betas > 0) & (schedule.betas < 1)).all().item())
        self.assertTrue(torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]).item())

    def test_cosine_schedule_is_finite_and_monotonic(self) -> None:
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=8, sampling_steps=4, schedule="cosine"))

        self.assertTrue(torch.isfinite(diffusion.schedule.alpha_bars).all().item())
        self.assertTrue(torch.all(diffusion.schedule.alpha_bars[1:] < diffusion.schedule.alpha_bars[:-1]).item())

    def test_q_sample_matches_formula(self) -> None:
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=4))
        x_0 = torch.ones(2, 3, 2, 2)
        noise = torch.full_like(x_0, 0.5)
        timesteps = torch.tensor([0, 3])

        actual = diffusion.q_sample(x_0, timesteps, noise)
        sqrt_alpha = diffusion.schedule.sqrt_alpha_bars[timesteps].reshape(2, 1, 1, 1)
        sqrt_one_minus = diffusion.schedule.sqrt_one_minus_alpha_bars[timesteps].reshape(2, 1, 1, 1)
        expected = sqrt_alpha * x_0 + sqrt_one_minus * noise

        self.assertTrue(torch.allclose(actual, expected))

    def test_training_loss_is_finite_scalar(self) -> None:
        torch.manual_seed(10)
        model = _tiny_model()
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=4))
        generator = torch.Generator().manual_seed(5)

        loss = diffusion.training_loss(model, torch.randn(2, 3, 64, 64), _conditions(2), generator)

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())

    def test_training_loss_cfg_dropout_supports_ddp_like_wrapper(self) -> None:
        torch.manual_seed(13)
        model = _DDPLikeWrapper(_tiny_model())
        diffusion = GaussianDiffusion(
            DiffusionConfig(train_timesteps=4, sampling_steps=4, cfg_dropout=1.0)
        )
        generator = torch.Generator().manual_seed(5)

        loss = diffusion.training_loss(model, torch.randn(2, 3, 64, 64), _conditions(2), generator)

        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())

    def test_sampling_is_repeatable_with_seeded_generator(self) -> None:
        torch.manual_seed(11)
        model = _tiny_model().eval()
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=3))

        first = diffusion.sample(
            model,
            _conditions(1),
            (1, 3, 64, 64),
            generator=torch.Generator().manual_seed(99),
        )
        second = diffusion.sample(
            model,
            _conditions(1),
            (1, 3, 64, 64),
            generator=torch.Generator().manual_seed(99),
        )

        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.isfinite(first).all().item())

    def test_sampling_supports_cfg_mixing(self) -> None:
        torch.manual_seed(12)
        model = _tiny_model().eval()
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=2))

        samples = diffusion.sample(
            model,
            _conditions(1),
            (1, 3, 64, 64),
            guidance_scale=1.5,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(tuple(samples.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(samples).all().item())

    def test_sampling_clips_predicted_clean_image(self) -> None:
        model = _ExtremeNoiseModel().eval()
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=4))

        samples = diffusion.sample(
            model,
            _conditions(1),
            (1, 3, 64, 64),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(torch.isfinite(samples).all().item())
        self.assertLessEqual(float(samples.max().item()), 1.0)
        self.assertGreaterEqual(float(samples.min().item()), -1.0)

    def test_sampling_cfg_mixing_supports_ddp_like_wrapper(self) -> None:
        torch.manual_seed(14)
        model = _DDPLikeWrapper(_tiny_model().eval())
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=2))

        samples = diffusion.sample(
            model,
            _conditions(1),
            (1, 3, 64, 64),
            guidance_scale=1.5,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(tuple(samples.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(samples).all().item())

    def test_tensor_to_uint8_images_clamps_and_reorders(self) -> None:
        samples = torch.tensor([[[[-2.0]], [[0.0]], [[2.0]]]])

        images = tensor_to_uint8_images(samples)

        self.assertEqual(images.dtype, torch.uint8)
        self.assertEqual(tuple(images.shape), (1, 1, 1, 3))
        self.assertEqual(images[0, 0, 0].tolist(), [0, 128, 255])

    def test_invalid_schedule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GaussianDiffusion(DiffusionConfig(schedule="bad"))

    def test_invalid_sampler_is_rejected(self) -> None:
        model = _tiny_model().eval()
        diffusion = GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=2))

        with self.assertRaises(ValueError):
            diffusion.sample(model, _conditions(1), (1, 3, 64, 64), sampler="ddim")

    def test_invalid_sampling_steps_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GaussianDiffusion(DiffusionConfig(train_timesteps=4, sampling_steps=5))


def _tiny_model() -> ConditionalUNet:
    return ConditionalUNet(
        UNetConfig(
            base_channels=4,
            channel_multipliers=(1,),
            embedding_dim=16,
            residual_blocks=1,
            dropout=0.0,
        )
    )


def _conditions(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "animal_id": torch.zeros(batch_size, dtype=torch.long),
        "object_id": torch.zeros(batch_size, dtype=torch.long),
        "pair_id": torch.zeros(batch_size, dtype=torch.long),
    }


class _DDPLikeWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _ExtremeNoiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_t, timesteps, animal_ids, object_ids, pair_ids):
        return torch.full_like(x_t, 1e6)


if __name__ == "__main__":
    unittest.main()
