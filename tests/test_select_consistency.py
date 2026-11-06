import unittest

import pandas as pd

from scripts.select_consistency import (
    select_configuration,
)


class ConsistencySelectionTests(unittest.TestCase):
    def test_near_tie_prefers_weaker_regularization(self):
        summary = pd.DataFrame(
            [
                {
                    "lambda": 0.5,
                    "rho": 0.0,
                    "mean_validation_score": 0.7000,
                    "eligible": True,
                },
                {
                    "lambda": 5.0,
                    "rho": 0.1,
                    "mean_validation_score": 0.7020,
                    "eligible": True,
                },
                {
                    "lambda": 10.0,
                    "rho": 0.2,
                    "mean_validation_score": 0.7100,
                    "eligible": False,
                },
            ]
        )

        selected = select_configuration(summary)

        self.assertEqual(float(selected["lambda"]), 0.5)
        self.assertEqual(float(selected["rho"]), 0.0)

    def test_no_eligible_configuration_is_rejected(self):
        summary = pd.DataFrame(
            [
                {
                    "lambda": 1.0,
                    "rho": 0.1,
                    "mean_validation_score": 0.8,
                    "eligible": False,
                }
            ]
        )

        with self.assertRaises(ValueError):
            select_configuration(summary)


if __name__ == "__main__":
    unittest.main()
