import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.transforms import (
    TRAIN_PADDING_MODES,
    random_translate_batch,
    translate_batch,
)
from src.evaluation.translation import (
    PRIMARY_PADDING_MODES,
    build_axial_shifts,
    evaluate_translation_robustness,
)


class ConstantClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = images.new_zeros((images.shape[0], 2))
        logits[:, 0] = 1.0
        return logits


class BoundaryModeTests(unittest.TestCase):
    def test_axial_shifts_follow_distances(self):
        self.assertEqual(
            build_axial_shifts([1, 2]),
            (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-2, 0), (2, 0), (0, -2), (0, 2),
            ),
        )

        with self.assertRaises(ValueError):
            build_axial_shifts([1, 1])

    def test_circular_translation_preserves_values(self):
        images = torch.arange(
            3 * 4 * 4,
            dtype=torch.float32,
        ).reshape(1, 3, 4, 4)

        shifted = translate_batch(
            images,
            dx=1,
            dy=-1,
            padding_mode="circular",
        )

        self.assertTrue(
            torch.equal(
                images.flatten().sort().values,
                shifted.flatten().sort().values,
            )
        )

    def test_random_boundary_translation_is_reproducible(self):
        images = torch.randn(12, 3, 8, 8)

        first, first_choices = random_translate_batch(
            images,
            generator=torch.Generator().manual_seed(7),
        )
        second, second_choices = random_translate_batch(
            images,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.equal(first_choices, second_choices)
        )
        self.assertEqual(
            TRAIN_PADDING_MODES,
            ("reflect", "replicate", "circular"),
        )

    def test_validation_uses_worst_padding_score(self):
        images = torch.randn(8, 3, 8, 8)
        targets = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(images, targets),
            batch_size=4,
        )

        metrics = evaluate_translation_robustness(
            ConstantClassifier(),
            loader,
            torch.device("cpu"),
        )

        self.assertEqual(
            PRIMARY_PADDING_MODES,
            ("reflect", "replicate", "circular"),
        )
        self.assertEqual(
            metrics["worst_padding_shifted_accuracy"],
            1.0,
        )
        self.assertEqual(metrics["validation_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
