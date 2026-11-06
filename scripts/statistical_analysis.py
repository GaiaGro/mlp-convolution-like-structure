from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t


ROOT = Path("results/final/evaluation")
PREDICTIONS = ROOT / "predictions"
OUTPUT = ROOT / "statistics"
SEEDS = [43, 44, 45, 46, 47]

COMPARISONS = [
    (
        "exposure_minus_baseline",
        ("mlp", "baseline"),
        ("mlp", "exposure"),
    ),
    (
        "consistency_minus_exposure",
        ("mlp", "exposure"),
        ("mlp", "consistency"),
    ),
    (
        "augmentation_minus_baseline",
        ("mlp", "baseline"),
        ("mlp", "augmentation"),
    ),
    (
        "augmentation_consistency_minus_consistency",
        ("mlp", "consistency"),
        ("mlp", "augmentation_consistency"),
    ),
    (
        "cnn_baseline_minus_mlp_baseline",
        ("mlp", "baseline"),
        ("cnn", "baseline"),
    ),
]

METRICS = [
    "clean_accuracy",
    "mean_shifted_accuracy",
    "worst_padding_shifted_accuracy",
    "distance_accuracy_auc",
    "worst_padding_distance_auc",
    "heldout_distance4_mean_accuracy",
    "heldout_distance4_worst_padding_accuracy",
    "mean_prediction_consistency",
    "worst_padding_prediction_consistency",
]


def select(runs, model, condition):
    result = runs[
        (runs["model"] == model)
        & (runs["condition"] == condition)
    ].copy()

    if set(result["seed"]) != set(SEEDS):
        raise ValueError(
            f"Incomplete seeds for {model}/{condition}."
        )

    if result["seed"].duplicated().any():
        raise ValueError(
            f"Duplicate seeds for {model}/{condition}."
        )

    return result.set_index("seed").sort_index()


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values)))
        * sorted_values
    ).clip(max=1.0)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def load_predictions(run_name):
    path = PREDICTIONS / f"{run_name}.npz"

    with np.load(path) as data:
        targets = data["targets"].copy()
        predictions = data["clean_predictions"].copy()

    if targets.shape != (10000,):
        raise ValueError(f"Unexpected shape in {path}.")

    return targets, predictions


def main():
    runs = pd.read_csv(ROOT / "test_runs.csv")

    if len(runs) != 35:
        raise ValueError(
            f"Expected 35 final runs, found {len(runs)}."
        )

    missing = set(
        METRICS
        + ["run_name", "model", "condition", "seed"]
    ) - set(runs.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    paired_rows = []
    paired_values = []
    mcnemar_rows = []

    critical = t.ppf(
        0.975,
        df=len(SEEDS) - 1,
    )

    for label, reference_key, comparison_key in COMPARISONS:
        reference = select(runs, *reference_key)
        comparison = select(runs, *comparison_key)

        for metric in METRICS:
            first = reference.loc[
                SEEDS, metric
            ].to_numpy(float)

            second = comparison.loc[
                SEEDS, metric
            ].to_numpy(float)

            differences = second - first
            mean_difference = differences.mean()
            sd_difference = differences.std(ddof=1)

            half_width = (
                critical
                * sd_difference
                / np.sqrt(len(SEEDS))
            )

            paired_rows.append({
                "comparison": label,
                "metric": metric,
                "n_seeds": len(SEEDS),
                "reference_mean": first.mean(),
                "comparison_mean": second.mean(),
                "mean_paired_difference": mean_difference,
                "sd_paired_difference": sd_difference,
                "ci95_lower": mean_difference - half_width,
                "ci95_upper": mean_difference + half_width,
            })

            for seed, a, b, difference in zip(
                SEEDS,
                first,
                second,
                differences,
            ):
                paired_values.append({
                    "comparison": label,
                    "metric": metric,
                    "seed": seed,
                    "reference_value": a,
                    "comparison_value": b,
                    "paired_difference": difference,
                })

        for seed in SEEDS:
            reference_name = reference.loc[
                seed, "run_name"
            ]
            comparison_name = comparison.loc[
                seed, "run_name"
            ]

            targets_a, predictions_a = load_predictions(
                reference_name
            )
            targets_b, predictions_b = load_predictions(
                comparison_name
            )

            if not np.array_equal(targets_a, targets_b):
                raise ValueError(
                    f"Target mismatch for {label}, seed {seed}."
                )

            correct_a = predictions_a == targets_a
            correct_b = predictions_b == targets_b

            reference_only = int(
                np.sum(correct_a & ~correct_b)
            )
            comparison_only = int(
                np.sum(~correct_a & correct_b)
            )

            discordant = reference_only + comparison_only
            p_value = 1.0

            if discordant:
                p_value = binomtest(
                    min(reference_only, comparison_only),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue

            mcnemar_rows.append({
                "comparison": label,
                "seed": seed,
                "reference_run": reference_name,
                "comparison_run": comparison_name,
                "n_images": len(targets_a),
                "reference_correct_comparison_wrong": (
                    reference_only
                ),
                "reference_wrong_comparison_correct": (
                    comparison_only
                ),
                "clean_accuracy_difference": (
                    comparison_only - reference_only
                ) / len(targets_a),
                "p_value_exact_two_sided": p_value,
                "holm_family": (
                    "5_comparisons_x_5_seeds_clean"
                ),
            })

    paired = pd.DataFrame(paired_rows)
    values = pd.DataFrame(paired_values)
    mcnemar = pd.DataFrame(mcnemar_rows)

    if len(mcnemar) != 25:
        raise AssertionError(
            "Expected 25 McNemar tests."
        )

    mcnemar["p_value_holm"] = holm_adjust(
        mcnemar["p_value_exact_two_sided"]
    )

    mcnemar["reject_h0_holm_0p05"] = (
        mcnemar["p_value_holm"] < 0.05
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    paired.to_csv(
        OUTPUT / "paired_seed_summary.csv",
        index=False,
    )
    values.to_csv(
        OUTPUT / "paired_seed_values.csv",
        index=False,
    )
    mcnemar.to_csv(
        OUTPUT / "mcnemar_clean_holm.csv",
        index=False,
    )

    headline = paired[
        paired["metric"].isin([
            "clean_accuracy",
            "mean_shifted_accuracy",
            "heldout_distance4_mean_accuracy",
        ])
    ].copy()

    for column in [
        "mean_paired_difference",
        "ci95_lower",
        "ci95_upper",
    ]:
        headline[column] *= 100

    print("PAIRED DIFFERENCES IN PERCENTAGE POINTS")

    print(
        headline[[
            "comparison",
            "metric",
            "mean_paired_difference",
            "ci95_lower",
            "ci95_upper",
        ]]
        .round(3)
        .to_string(index=False)
    )

    print("\nEXACT McNEMAR, GLOBAL HOLM CORRECTION")

    print(
        mcnemar.groupby("comparison")
        .agg(
            seeds=("seed", "count"),
            significant_seeds=(
                "reject_h0_holm_0p05",
                "sum",
            ),
            maximum_adjusted_p=(
                "p_value_holm",
                "max",
            ),
        )
        .reset_index()
        .round(6)
        .to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
