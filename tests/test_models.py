import unittest

import torch
from torch import nn

from src.models import (
    ResidualMLP,
    SmallCNN,
    build_model,
    count_parameters,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.randn(4, 3, 32, 32)
        self.mlp = ResidualMLP()
        self.cnn = SmallCNN()

    def test_output_shapes(self):
        self.assertEqual(
            self.mlp(self.images).shape,
            (4, 10),
        )
        self.assertEqual(
            self.cnn(self.images).shape,
            (4, 10),
        )

    def test_feature_shapes(self):
        self.assertEqual(
            self.mlp.forward_features(self.images).shape,
            (4, 512),
        )
        self.assertEqual(
            self.cnn.forward_features(self.images).shape,
            (4, 384),
        )

    def test_mlp_contains_no_convolution(self):
        convolutions = [
            module
            for module in self.mlp.modules()
            if isinstance(module, nn.Conv2d)
        ]
        self.assertEqual(convolutions, [])

    def test_parameter_counts_are_matched(self):
        mlp_parameters = count_parameters(self.mlp)
        cnn_parameters = count_parameters(self.cnn)

        relative_difference = abs(
            mlp_parameters - cnn_parameters
        ) / mlp_parameters

        self.assertLessEqual(relative_difference, 0.10)

    def test_invalid_image_size_is_rejected(self):
        with self.assertRaises(ValueError):
            self.mlp(torch.randn(2, 3, 28, 28))


    def test_pixel_permutation_is_fixed_and_complete(self):
        first = ResidualMLP(pixel_permutation_seed=0)
        second = ResidualMLP(pixel_permutation_seed=0)
        third = ResidualMLP(pixel_permutation_seed=1)

        self.assertTrue(
            torch.equal(
                first.pixel_permutation,
                second.pixel_permutation,
            )
        )
        self.assertFalse(
            torch.equal(
                first.pixel_permutation,
                third.pixel_permutation,
            )
        )
        self.assertEqual(
            torch.unique(first.pixel_permutation).numel(),
            32 * 32,
        )

    def test_pixel_permutation_preserves_interface(self):
        config = {
            "mlp": {
                "hidden_width": 512,
                "residual_blocks": 3,
                "expansion_ratio": 2,
                "dropout": 0.1,
            }
        }

        model = build_model(
            "mlp",
            config,
            pixel_permutation_seed=0,
        )

        logits, features = model.forward_with_features(
            self.images
        )

        self.assertEqual(logits.shape, (4, 10))
        self.assertEqual(features.shape, (4, 512))
        self.assertEqual(
            count_parameters(model),
            count_parameters(self.mlp),
        )
        self.assertNotIn(
            "pixel_permutation",
            model.state_dict(),
        )

    def test_pixel_permutation_preserves_rgb_triplets(self):
        model = ResidualMLP(pixel_permutation_seed=0)

        images = torch.arange(
            2 * 3 * 32 * 32,
            dtype=torch.float32,
        ).reshape(2, 3, 32, 32)

        captured = {}

        def capture_input(_module, inputs):
            captured["flattened"] = (
                inputs[0].detach().clone()
            )

        handle = (
            model.input_projection
            .register_forward_pre_hook(capture_input)
        )

        model(images)
        handle.remove()

        expected = (
            images.flatten(start_dim=2)
            .index_select(
                2,
                model.pixel_permutation,
            )
            .reshape_as(images)
            .flatten(start_dim=1)
        )

        self.assertTrue(
            torch.equal(
                captured["flattened"],
                expected,
            )
        )



if __name__ == "__main__":
    unittest.main()
