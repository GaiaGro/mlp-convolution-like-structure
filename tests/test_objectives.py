import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.training.objectives import (
    CONDITIONS,
    TWO_VIEW_CONDITIONS,
    compute_training_objective,
    jensen_shannon_divergence,
    variance_covariance_regularization,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(3 * 8 * 8, 16)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(16, 10)

    def forward_features(self, images):
        return self.dropout(
            self.activation(
                self.feature_layer(self.flatten(images))
            )
        )

    def forward_with_features(self, images):
        features = self.forward_features(images)
        return self.classifier(features), features

    def forward(self, images):
        logits, _ = self.forward_with_features(images)
        return logits


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.randn(8, 3, 8, 8)
        self.targets = torch.arange(8) % 10
        self.criterion = nn.CrossEntropyLoss()

    def test_js_is_zero_for_identical_logits(self):
        logits = torch.randn(12, 10)
        divergence = jensen_shannon_divergence(
            logits,
            logits,
        )
        self.assertAlmostEqual(
            divergence.item(),
            0.0,
            places=6,
        )

    def test_js_is_symmetric_and_non_negative(self):
        first = torch.randn(12, 10)
        second = torch.randn(12, 10)

        forward = jensen_shannon_divergence(first, second)
        backward = jensen_shannon_divergence(second, first)

        self.assertGreaterEqual(forward.item(), -1e-7)
        torch.testing.assert_close(forward, backward)

    def test_vicreg_penalizes_collapsed_features(self):
        collapsed = torch.zeros(8, 16)
        combined, variance, covariance = (
            variance_covariance_regularization(
                collapsed,
                collapsed,
            )
        )

        self.assertGreater(variance.item(), 0.9)
        self.assertEqual(covariance.item(), 0.0)
        self.assertTrue(torch.isfinite(combined).item())

    def test_vicreg_has_finite_gradients(self):
        first = torch.randn(
            8,
            16,
            requires_grad=True,
        )
        second = torch.randn(
            8,
            16,
            requires_grad=True,
        )

        combined, _, _ = (
            variance_covariance_regularization(
                first,
                second,
            )
        )
        combined.backward()

        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_every_condition_has_finite_gradients(self):
        for condition in sorted(CONDITIONS):
            with self.subTest(condition=condition):
                model = TinyClassifier()
                consistency_weight = (
                    2.0
                    if condition
                    in {
                        "consistency",
                        "augmentation_consistency",
                    }
                    else 0.0
                )
                variance_weight = (
                    0.1
                    if condition
                    in {
                        "consistency",
                        "augmentation_consistency",
                    }
                    else 0.0
                )

                output = compute_training_objective(
                    model=model,
                    images=self.images,
                    targets=self.targets,
                    criterion=self.criterion,
                    condition=condition,
                    consistency_weight=consistency_weight,
                    variance_covariance_weight=variance_weight,
                    translation_generator=(
                        torch.Generator().manual_seed(3)
                    ),
                )

                self.assertTrue(
                    torch.isfinite(output.loss).item()
                )
                output.loss.backward()

                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(
                        torch.isfinite(parameter.grad).all()
                    )

                self.assertEqual(
                    output.translated_logits is not None,
                    condition in TWO_VIEW_CONDITIONS,
                )

    def test_exposure_has_no_regularization_penalty(self):
        model = TinyClassifier()
        output = compute_training_objective(
            model=model,
            images=self.images,
            targets=self.targets,
            criterion=self.criterion,
            condition="exposure",
            translation_generator=(
                torch.Generator().manual_seed(4)
            ),
        )

        self.assertEqual(
            output.consistency_loss.item(),
            0.0,
        )
        self.assertEqual(
            output.variance_covariance_loss.item(),
            0.0,
        )

    def test_paired_views_share_dropout_mask(self):
        model = TinyClassifier()
        model.train()

        with patch(
            "src.training.objectives.random_translate_batch",
            return_value=(
                self.images.clone(),
                torch.zeros(8, dtype=torch.long),
            ),
        ):
            output = compute_training_objective(
                model=model,
                images=self.images,
                targets=self.targets,
                criterion=self.criterion,
                condition="consistency",
                consistency_weight=1.0,
                translation_generator=(
                    torch.Generator().manual_seed(5)
                ),
            )

        torch.testing.assert_close(
            output.original_logits,
            output.translated_logits,
        )
        self.assertAlmostEqual(
            output.consistency_loss.item(),
            0.0,
            places=7,
        )

    def test_invalid_weights_are_rejected(self):
        for keyword in (
            "consistency_weight",
            "variance_covariance_weight",
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    compute_training_objective(
                        model=TinyClassifier(),
                        images=self.images,
                        targets=self.targets,
                        criterion=self.criterion,
                        condition="baseline",
                        **{keyword: 1.0},
                    )


if __name__ == "__main__":
    unittest.main()
