from __future__ import annotations

import unittest
from unittest import mock

import torch

from scripts.model import ConditionalUNet, UNetConfig, sinusoidal_timestep_embedding


class ConditionalUNetTests(unittest.TestCase):
    def test_forward_preserves_image_shape(self) -> None:
        torch.manual_seed(1)
        model = _tiny_model()
        x_t, timesteps, animal_ids, object_ids, pair_ids = _batch()

        output = model(x_t, timesteps, animal_ids, object_ids, pair_ids)

        self.assertEqual(tuple(output.shape), tuple(x_t.shape))

    def test_timestep_changes_output(self) -> None:
        torch.manual_seed(2)
        model = _tiny_model().eval()
        x_t, _, animal_ids, object_ids, pair_ids = _batch()

        first = model(x_t, torch.tensor([1, 1]), animal_ids, object_ids, pair_ids)
        second = model(x_t, torch.tensor([7, 7]), animal_ids, object_ids, pair_ids)

        self.assertFalse(torch.allclose(first, second))

    def test_condition_changes_output(self) -> None:
        torch.manual_seed(3)
        model = _tiny_model().eval()
        x_t, timesteps, animal_ids, object_ids, pair_ids = _batch()

        first = model(x_t, timesteps, animal_ids, object_ids, pair_ids)
        second = model(
            x_t,
            timesteps,
            torch.tensor([1, 1]),
            torch.tensor([1, 1]),
            torch.tensor([11, 11]),
        )

        self.assertFalse(torch.allclose(first, second))

    def test_backward_produces_finite_gradients(self) -> None:
        torch.manual_seed(4)
        model = _tiny_model()
        x_t, timesteps, animal_ids, object_ids, pair_ids = _batch()

        loss = model(x_t, timesteps, animal_ids, object_ids, pair_ids).square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]

        self.assertTrue(gradients)
        self.assertTrue(any(torch.isfinite(gradient).all().item() for gradient in gradients))

    def test_attention_preserves_image_shape(self) -> None:
        torch.manual_seed(5)
        model = ConditionalUNet(
            UNetConfig(
                base_channels=8,
                channel_multipliers=(1, 2),
                embedding_dim=32,
                residual_blocks=1,
                attention_resolutions=(32,),
                attention_heads=2,
            )
        )
        x_t, timesteps, animal_ids, object_ids, pair_ids = _batch()

        output = model(x_t, timesteps, animal_ids, object_ids, pair_ids)

        self.assertEqual(tuple(output.shape), tuple(x_t.shape))
        self.assertTrue(any("qkv" in name for name, _ in model.named_parameters()))

    def test_invalid_image_shape_is_rejected(self) -> None:
        model = _tiny_model()
        x_t, timesteps, animal_ids, object_ids, pair_ids = _batch()

        with self.assertRaises(ValueError):
            model(x_t[:, :, :32, :32], timesteps, animal_ids, object_ids, pair_ids)

    def test_invalid_label_range_is_rejected(self) -> None:
        model = _tiny_model()
        x_t, timesteps, animal_ids, object_ids, pair_ids = _batch()

        with self.assertRaises(ValueError):
            model(x_t, timesteps, torch.tensor([0, model.config.num_animals + 1]), object_ids, pair_ids)

    def test_non_integer_labels_are_rejected(self) -> None:
        model = _tiny_model()
        x_t, timesteps, _, object_ids, pair_ids = _batch()

        with self.assertRaises(ValueError):
            model(x_t, timesteps, torch.tensor([0.0, 1.0]), object_ids, pair_ids)

    def test_cfg_null_ids_are_accepted_when_enabled(self) -> None:
        model = _tiny_model()
        x_t, timesteps, _, _, _ = _batch()
        null_ids = model.null_condition_ids(batch_size=2, device=x_t.device)

        output = model(
            x_t,
            timesteps,
            null_ids["animal_id"],
            null_ids["object_id"],
            null_ids["pair_id"],
        )

        self.assertEqual(tuple(output.shape), tuple(x_t.shape))

    def test_constructor_does_not_load_checkpoints(self) -> None:
        with mock.patch("torch.load", side_effect=AssertionError("checkpoint load not allowed")):
            model = _tiny_model()

        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_sinusoidal_timestep_embedding_shape(self) -> None:
        embedding = sinusoidal_timestep_embedding(torch.tensor([0, 1, 2]), embedding_dim=7)

        self.assertEqual(tuple(embedding.shape), (3, 7))
        self.assertTrue(torch.isfinite(embedding).all().item())

    def test_invalid_attention_resolution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConditionalUNet(UNetConfig(attention_resolutions=(7,)))


def _tiny_model() -> ConditionalUNet:
    return ConditionalUNet(
        UNetConfig(
            base_channels=8,
            channel_multipliers=(1, 2),
            embedding_dim=32,
            residual_blocks=1,
            dropout=0.0,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(2, 3, 64, 64),
        torch.tensor([1, 5]),
        torch.tensor([0, 8]),
        torch.tensor([0, 9]),
        torch.tensor([0, 89]),
    )


if __name__ == "__main__":
    unittest.main()
