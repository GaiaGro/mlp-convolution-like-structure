import unittest

import torch

from scripts.representation_diagnostics import (
    representation_statistics,
)


class RepresentationDiagnosticTests(unittest.TestCase):
    def test_diverse_features_exceed_collapsed_rank(self):
        collapsed = torch.zeros(16, 8)
        diverse = torch.eye(8).repeat(2, 1)

        collapsed_rank, _, _, _ = (
            representation_statistics(collapsed)
        )
        diverse_rank, _, _, _ = (
            representation_statistics(diverse)
        )

        self.assertEqual(collapsed_rank, 0.0)
        self.assertAlmostEqual(diverse_rank, 7.0, places=10)

    def test_statistics_are_finite(self):
        features = torch.randn(32, 12)
        rank, variances, mean_std, total_variance = (
            representation_statistics(features)
        )

        self.assertGreater(rank, 0)
        self.assertTrue(torch.isfinite(variances).all())
        self.assertTrue(math_is_finite(mean_std))
        self.assertTrue(math_is_finite(total_variance))


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


if __name__ == "__main__":
    unittest.main()
