import unittest

import torch

from src.evaluation.structural import (
    extract_maximum_energy_patches,
    cross_validated_template_reuse,
    fit_sign_invariant_spherical_kmeans,
    maximum_window_energy_fraction,
    shuffle_spatial_weights,
    weights_in_original_image_coordinates,
)


class StructuralMetricTests(unittest.TestCase):
    def test_localized_weights_have_greater_locality(self):
        generator = torch.Generator().manual_seed(1)

        diffuse = torch.randn(
            4,
            3,
            32,
            32,
            generator=generator,
        )

        localized = torch.zeros(4, 3, 32, 32)
        localized[:, :, 10:17, 12:19] = torch.randn(
            4,
            3,
            7,
            7,
            generator=generator,
        )

        diffuse_score, _ = maximum_window_energy_fraction(
            diffuse,
            7,
        )
        localized_score, _ = maximum_window_energy_fraction(
            localized,
            7,
        )

        self.assertTrue(
            torch.allclose(
                localized_score,
                torch.ones_like(localized_score),
            )
        )
        self.assertGreater(
            localized_score.mean().item(),
            diffuse_score.mean().item(),
        )

    def test_maximum_energy_patch_is_located_correctly(self):
        weights = torch.zeros(1, 3, 10, 10)
        weights[:, :, 2:5, 4:7] = 1.0

        patches, fractions, locations = (
            extract_maximum_energy_patches(weights, 3)
        )

        self.assertEqual(tuple(patches.shape), (1, 3, 3, 3))
        self.assertEqual(locations.tolist(), [[2, 4]])
        self.assertAlmostEqual(fractions.item(), 1.0)

    def test_spatial_shuffle_preserves_rgb_triplets(self):
        positions = torch.arange(
            1,
            17,
            dtype=torch.float32,
        )
        weights = torch.stack(
            (
                positions,
                positions + 100,
                positions + 200,
            )
        ).reshape(1, 3, 4, 4)

        first = shuffle_spatial_weights(
            weights,
            torch.Generator().manual_seed(3),
        )
        second = shuffle_spatial_weights(
            weights,
            torch.Generator().manual_seed(3),
        )

        flattened = first.reshape(1, 3, 16)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.equal(
                flattened[:, 1] - flattened[:, 0],
                torch.full((1, 16), 100.0),
            )
        )
        self.assertTrue(
            torch.equal(
                flattened[:, 2] - flattened[:, 0],
                torch.full((1, 16), 200.0),
            )
        )
        self.assertTrue(
            torch.equal(
                flattened[:, 0].sort().values,
                positions.reshape(1, 16),
            )
        )

    def test_pixel_permutation_maps_back_exactly(self):
        weights = torch.tensor(
            [
                [
                    [[10.0, 20.0], [30.0, 40.0]],
                    [[11.0, 21.0], [31.0, 41.0]],
                    [[12.0, 22.0], [32.0, 42.0]],
                ]
            ]
        )
        permutation = torch.tensor([2, 0, 3, 1])

        remapped = weights_in_original_image_coordinates(
            weights,
            permutation,
        )

        expected_first_channel = torch.tensor(
            [[20.0, 40.0], [10.0, 30.0]]
        )

        self.assertTrue(
            torch.equal(
                remapped[0, 0],
                expected_first_channel,
            )
        )

    def test_invalid_permutation_is_rejected(self):
        weights = torch.ones(1, 3, 2, 2)

        with self.assertRaises(ValueError):
            weights_in_original_image_coordinates(
                weights,
                torch.tensor([0, 1, 1, 3]),
            )


    def test_template_fitting_is_sign_and_scale_invariant(self):
        samples = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-2.0, -4.0, -6.0],
                [0.0, 1.0, 0.0],
                [0.0, -3.0, 0.0],
            ]
        )

        templates = fit_sign_invariant_spherical_kmeans(
            samples=samples,
            template_count=2,
            initializations=5,
            maximum_iterations=100,
            tolerance=1e-6,
            generator=torch.Generator().manual_seed(4),
        )

        normalized = samples / samples.norm(
            dim=1,
            keepdim=True,
        )
        matches = (
            normalized @ templates.T
        ).abs().max(dim=1).values

        self.assertTrue(
            torch.allclose(
                matches,
                torch.ones_like(matches),
                atol=1e-6,
            )
        )



    def test_translated_template_reuse_recovers_known_bank(self):
        generator = torch.Generator().manual_seed(8)
        template = torch.randn(
            3,
            3,
            3,
            generator=generator,
        )

        structured = torch.zeros(12, 3, 8, 8)

        for neuron in range(12):
            row = neuron % 6
            column = (2 * neuron) % 6
            scale = (neuron + 1) / 4

            if neuron % 2:
                scale = -scale

            structured[
                neuron,
                :,
                row : row + 3,
                column : column + 3,
            ] = scale * template

        scores = cross_validated_template_reuse(
            weights=structured,
            patch_size=3,
            template_count=1,
            folds=3,
            repetitions=2,
            initializations=2,
            maximum_iterations=50,
            tolerance=1e-6,
            seed=9,
        )

        self.assertTrue(
            torch.all(
                scores[
                    "conditional_shape_score"
                ] > 0.999
            )
        )
        self.assertTrue(
            torch.all(
                scores[
                    "total_energy_reconstruction_fraction"
                ] > 0.999
            )
        )

    def test_template_reuse_distinguishes_random_weights(self):
        generator = torch.Generator().manual_seed(10)
        template = torch.randn(
            3,
            3,
            3,
            generator=generator,
        )

        structured = torch.zeros(12, 3, 8, 8)

        for neuron in range(12):
            row = neuron % 6
            column = (3 * neuron) % 6
            structured[
                neuron,
                :,
                row : row + 3,
                column : column + 3,
            ] = template

        random_weights = torch.randn(
            12,
            3,
            8,
            8,
            generator=generator,
        )

        arguments = {
            "patch_size": 3,
            "template_count": 1,
            "folds": 3,
            "repetitions": 2,
            "initializations": 2,
            "maximum_iterations": 50,
            "tolerance": 1e-6,
            "seed": 11,
        }

        structured_scores = (
            cross_validated_template_reuse(
                weights=structured,
                **arguments,
            )
        )
        random_scores = cross_validated_template_reuse(
            weights=random_weights,
            **arguments,
        )

        structured_mean = structured_scores[
            "total_energy_reconstruction_fraction"
        ].mean()
        random_mean = random_scores[
            "total_energy_reconstruction_fraction"
        ].mean()

        self.assertGreater(
            structured_mean.item(),
            random_mean.item(),
        )

    def test_template_reuse_is_reproducible(self):
        generator = torch.Generator().manual_seed(12)
        weights = torch.randn(
            12,
            3,
            8,
            8,
            generator=generator,
        )

        arguments = {
            "weights": weights,
            "patch_size": 3,
            "template_count": 2,
            "folds": 3,
            "repetitions": 2,
            "initializations": 2,
            "maximum_iterations": 50,
            "tolerance": 1e-6,
            "seed": 13,
        }

        first = cross_validated_template_reuse(
            **arguments
        )
        second = cross_validated_template_reuse(
            **arguments
        )

        for key in first:
            self.assertTrue(
                torch.equal(first[key], second[key])
            )



if __name__ == "__main__":
    unittest.main()
