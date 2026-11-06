from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


INPUT = Path(
    "results/structural/analysis/merged/structural_analysis_all.csv"
)
OUTPUT_DIR = Path(
    "results/structural/analysis/statistics"
)

SEEDS = [43, 44, 45, 46, 47]

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

PRIMARY_METRICS = [
    "mean_locality_window_7",
    "mean_template_total_reconstruction",
]

COMPARISONS = [
    "random_initialization",
    "spatial_shuffle",
    "pixel_permutation_model",
    "pixel_permutation_original_image",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_input(df: pd.DataFrame) -> None:
    require(
        len(df) == 580,
        f"Expected 580 rows, found {len(df)}.",
    )

    require(
        df["analysis_id"].is_unique,
        "analysis_id contains duplicates.",
    )

    require(
        set(df["seed"]) == set(SEEDS),
        f"Unexpected seeds: {sorted(df['seed'].unique())}",
    )

    required_columns = {
        "analysis_id",
        "source",
        "coordinate_system",
        "condition",
        "seed",
        "shuffle_index",
        *PRIMARY_METRICS,
    }

    missing = required_columns - set(df.columns)
    require(
        not missing,
        f"Missing required columns: {sorted(missing)}",
    )

    require(
        not df[PRIMARY_METRICS].isna().any().any(),
        "Primary metrics contain missing values.",
    )

    expected_source_counts = {
        "trained": 25,
        "spatial_shuffle": 500,
        "pixel_permutation": 50,
        "random_initialization": 5,
    }

    observed_source_counts = (
        df["source"].value_counts().to_dict()
    )

    require(
        observed_source_counts == expected_source_counts,
        "Unexpected source counts: "
        f"{observed_source_counts}",
    )

    for seed in SEEDS:
        for condition in CONDITIONS:
            trained = df[
                (df["source"] == "trained")
                & (df["seed"] == seed)
                & (df["condition"] == condition)
            ]
            require(
                len(trained) == 1,
                f"Expected 1 trained row for "
                f"{condition}, seed {seed}; found {len(trained)}.",
            )

            shuffled = df[
                (df["source"] == "spatial_shuffle")
                & (df["seed"] == seed)
                & (df["condition"] == condition)
            ]
            require(
                len(shuffled) == 20,
                f"Expected 20 shuffled rows for "
                f"{condition}, seed {seed}; found {len(shuffled)}.",
            )

            require(
                shuffled["shuffle_index"].nunique() == 20,
                f"Shuffle indices are not unique for "
                f"{condition}, seed {seed}.",
            )

            pixel_model = df[
                (df["source"] == "pixel_permutation")
                & (df["coordinate_system"] == "model")
                & (df["seed"] == seed)
                & (df["condition"] == condition)
            ]
            require(
                len(pixel_model) == 1,
                f"Expected 1 model-coordinate pixel row for "
                f"{condition}, seed {seed}; "
                f"found {len(pixel_model)}.",
            )

            pixel_image = df[
                (df["source"] == "pixel_permutation")
                & (df["coordinate_system"] == "original_image")
                & (df["seed"] == seed)
                & (df["condition"] == condition)
            ]
            require(
                len(pixel_image) == 1,
                f"Expected 1 original-image pixel row for "
                f"{condition}, seed {seed}; "
                f"found {len(pixel_image)}.",
            )

    random_rows = df[
        df["source"] == "random_initialization"
    ]

    require(
        len(random_rows) == 5,
        f"Expected 5 random-initialization rows, "
        f"found {len(random_rows)}.",
    )

    require(
        not random_rows["seed"].duplicated().any(),
        "Random-initialization rows contain duplicate seeds.",
    )

    require(
        set(random_rows["condition"]) == {"not_applicable"},
        "Random-initialization condition is unexpected.",
    )


def build_seed_level_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    random_by_seed = (
        df[df["source"] == "random_initialization"]
        .set_index("seed")
        .sort_index()
    )

    for condition in CONDITIONS:
        for seed in SEEDS:
            trained = df[
                (df["source"] == "trained")
                & (df["condition"] == condition)
                & (df["seed"] == seed)
            ].iloc[0]

            shuffled = df[
                (df["source"] == "spatial_shuffle")
                & (df["condition"] == condition)
                & (df["seed"] == seed)
            ]

            pixel_model = df[
                (df["source"] == "pixel_permutation")
                & (df["coordinate_system"] == "model")
                & (df["condition"] == condition)
                & (df["seed"] == seed)
            ].iloc[0]

            pixel_image = df[
                (df["source"] == "pixel_permutation")
                & (df["coordinate_system"] == "original_image")
                & (df["condition"] == condition)
                & (df["seed"] == seed)
            ].iloc[0]

            random_row = random_by_seed.loc[seed]

            for metric in PRIMARY_METRICS:
                trained_value = float(trained[metric])

                controls = {
                    "random_initialization": float(
                        random_row[metric]
                    ),
                    "spatial_shuffle": float(
                        shuffled[metric].mean()
                    ),
                    "pixel_permutation_model": float(
                        pixel_model[metric]
                    ),
                    "pixel_permutation_original_image": float(
                        pixel_image[metric]
                    ),
                }

                for comparison, control_value in controls.items():
                    rows.append(
                        {
                            "condition": condition,
                            "metric": metric,
                            "comparison": comparison,
                            "seed": seed,
                            "trained_value": trained_value,
                            "control_value": control_value,
                            "paired_difference": (
                                trained_value - control_value
                            ),
                        }
                    )

    result = pd.DataFrame(rows)

    expected_rows = (
        len(CONDITIONS)
        * len(PRIMARY_METRICS)
        * len(COMPARISONS)
        * len(SEEDS)
    )

    require(
        len(result) == expected_rows,
        f"Expected {expected_rows} seed-level rows, "
        f"found {len(result)}.",
    )

    return result


def summarize_paired_differences(
    seed_values: pd.DataFrame,
) -> pd.DataFrame:
    critical = t.ppf(
        0.975,
        df=len(SEEDS) - 1,
    )

    rows = []

    grouped = seed_values.groupby(
        ["condition", "metric", "comparison"],
        sort=False,
    )

    for (condition, metric, comparison), group in grouped:
        group = group.sort_values("seed")

        require(
            group["seed"].tolist() == SEEDS,
            f"Incomplete seed pairing for "
            f"{condition}, {metric}, {comparison}.",
        )

        differences = group[
            "paired_difference"
        ].to_numpy(dtype=float)

        mean_difference = float(differences.mean())
        sd_difference = float(differences.std(ddof=1))

        standard_error = (
            sd_difference / np.sqrt(len(SEEDS))
        )

        half_width = critical * standard_error

        rows.append(
            {
                "condition": condition,
                "metric": metric,
                "comparison": comparison,
                "n_seeds": len(SEEDS),
                "trained_mean": float(
                    group["trained_value"].mean()
                ),
                "control_mean": float(
                    group["control_value"].mean()
                ),
                "mean_paired_difference": mean_difference,
                "sd_paired_difference": sd_difference,
                "ci95_lower": mean_difference - half_width,
                "ci95_upper": mean_difference + half_width,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    validate_input(df)

    seed_values = build_seed_level_table(df)

    summary = summarize_paired_differences(
        seed_values
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_values.to_csv(
        OUTPUT_DIR / "structural_paired_seed_values.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "structural_paired_summary.csv",
        index=False,
    )

    print("INPUT VALIDATION: OK")
    print()
    print(
        summary[
            [
                "condition",
                "metric",
                "comparison",
                "mean_paired_difference",
                "ci95_lower",
                "ci95_upper",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print()
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
