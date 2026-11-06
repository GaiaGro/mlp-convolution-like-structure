import unittest

import numpy as np
import torch

from src.data.cifar10 import SplitSpec, stratified_split_indices
from src.data.transforms import (
    TRAIN_SHIFTS,
    random_translate_batch,
    translate_batch,
)


class SplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_stratified(self):
        targets = np.repeat(np.arange(10), 500)
        specification = SplitSpec(
            train_size=4500,
            validation_size=500,
            seed=2026,
        )

        first = stratified_split_indices(targets, specification)
        second = stratified_split_indices(targets, specification)

        expected_sizes = {
            "train": 4500,
            "validation": 500,
        }

        for split, expected_size in expected_sizes.items():
            np.testing.assert_array_equal(
                first[split],
                second[split],
            )
            self.assertEqual(len(first[split]), expected_size)

            counts = np.bincount(
                targets[first[split]],
                minlength=10,
            )
            self.assertTrue(
                np.all(counts == expected_size // 10)
            )

        combined = np.concatenate(list(first.values()))
        self.assertEqual(
            len(np.unique(combined)),
            len(targets),
        )


class TranslationTests(unittest.TestCase):
    def test_translation_direction(self):
        image = torch.arange(
            25,
            dtype=torch.float32,
        ).reshape(1, 1, 5, 5)

        right = translate_batch(image, dx=1, dy=0)
        down = translate_batch(image, dx=0, dy=1)

        self.assertEqual(
            right[0, 0, 2, 3].item(),
            image[0, 0, 2, 2].item(),
        )
        self.assertEqual(
            down[0, 0, 3, 2].item(),
            image[0, 0, 2, 2].item(),
        )

    def test_random_translation_is_reproducible(self):
        images = torch.randn(16, 3, 8, 8)

        generator_a = torch.Generator().manual_seed(99)
        generator_b = torch.Generator().manual_seed(99)

        translated_a, choices_a = random_translate_batch(
            images,
            generator=generator_a,
        )
        translated_b, choices_b = random_translate_batch(
            images,
            generator=generator_b,
        )

        torch.testing.assert_close(
            translated_a,
            translated_b,
        )
        torch.testing.assert_close(choices_a, choices_b)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            translate_batch(
                torch.zeros(3, 32, 32),
                dx=1,
                dy=0,
            )


if __name__ == "__main__":
    unittest.main()
