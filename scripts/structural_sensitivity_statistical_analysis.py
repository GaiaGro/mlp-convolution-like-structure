from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


INPUT = Path(
    "results/structural/sensitivity/structural_sensitivity_all.csv"
)

OUTPUT_DIR = Path(
    "results/structural/sensitivity/statistics"
)

SEEDS = [43, 44, 45, 46, 47]

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

VARIANTS = [
    "patch5_templates16",
    "patch9_templates16",
    "patch7_templates8",
    "patch7_templates32",
]

METRIC = "mean_template_total_reconstruction"

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
        len(df) == 2320,
        f"Expected 2320 rows, found {len(df)}.",
    )

    require(
        not df.duplicated(
            subset=["sensitivity_id", "analysis_id"]
        ).any(),
        "Duplicate sensitivity_id + analysis_id pairs found.",
    )

    require(
        set(df["seed"]) == set(SEEDS),
        f"Unexpected seeds: {sorted(df['seed'].unique())}",
    )

    require(
        set(df["sensitivity_id"]) == set(VARIANTS),
        "Unexpected sensitivity variants.",
    )

    require(
        not df[METRIC].isna().any(),
        f"{METRIC} contains missing values.",
    )

    for variant in VARIANTS:
        variant_df = df[
            df["sensitivity_id"] == variant
        ]

        require(
            len(variant_df) == 580,
            f"{variant}: expected 580 rows, found {len(variant_df)}.",
        )

        for seed in SEEDS:
            for condition in CONDITIONS:
                trained = variant_df[
                    (variant_df["source"] == "trained")
                    & (variant_df["seed"] == seed)
                    & (variant_df["condition"] == condition)
                ]

                require(
                    len(trained) == 1,
                    f"{variant}, {condition}, seed {seed}: "
                    f"expected 1 trained row.",
                )

                shuffled = variant_df[
                    (variant_df["source"] == "spatial_shuffle")
                    & (variant_df["seed"] == seed)
                    & (variant_df["condition"] == condition)
                ]

                require(
                    len(shuffled) == 20,
                    f"{variant}, {condition}, seed {seed}: "
                    f"expected 20 shuffle rows.",
                )

                require(
                    shuffled["shuffle_index"].nunique() == 20,
                    f"{variant}, {condition}, seed {seed}: "
                    f"shuffle indices are not unique.",
                )

                pixel_model = variant_df[
                    (variant_df["source"] == "pixel_permutation")
                    & (variant_df["coordinate_system"] == "model")
                    & (variant_df["seed"] == seed)
                    & (variant_df["condition"] == condition)
                ]

                require(
                    len(pixel_model) == 1,
                    f"{variant}, {condition}, seed {seed}: "
                    f"expected 1 pixel-model row.",
                )

                pixel_image = variant_df[
                    (variant_df["source"] == "pixel_permutation")
                    & (
                        variant_df["coordinate_system"]
                        == "original_image"
                    )
                    & (variant_df["seed"] == seed)
                    & (variant_df["condition"] == condition)
                ]

                require(
                    len(pixel_image) == 1,
                    f"{variant}, {condition}, seed {seed}: "
                    f"expected 1 pixel-original row.",
                )

        random_rows = variant_df[
            variant_df["source"] == "random_initialization"
        ]

        require(
            len(random_rows) == 5,
            f"{variant}: expected 5 random rows.",
        )

        require(
            not random_rows["seed"].duplicated().any(),
            f"{variant}: duplicate random-initialization seeds.",
        )


def build_seed_level_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for variant in VARIANTS:
        variant_df = df[
            df["sensitivity_id"] == variant
        ]

        random_by_seed = (
            variant_df[
                variant_df["source"]
                == "random_initialization"
            ]
            .set_index("seed")
            .sort_index()
        )

        for condition in CONDITIONS:
            for seed in SEEDS:
                trained = variant_df[
                    (variant_df["source"] == "trained")
                    & (variant_df["condition"] == condition)
                    & (variant_df["seed"] == seed)
                ].iloc[0]

                shuffled = variant_df[
                    (
                        variant_df["source"]
                        == "spatial_shuffle"
                    )
                    & (variant_df["condition"] == condition)
                    & (variant_df["seed"] == seed)
                ]

                pixel_model = variant_df[
                    (
                        variant_df["source"]
                        == "pixel_permutation"
                    )
                    & (
                        variant_df["coordinate_system"]
                        == "model"
                    )
                    & (variant_df["condition"] == condition)
                    & (variant_df["seed"] == seed)
                ].iloc[0]

                pixel_image = variant_df[
                    (
                        variant_df["source"]
                        == "pixel_permutation"
                    )
                    & (
                        variant_df["coordinate_system"]
                        == "original_image"
                    )
                    & (variant_df["condition"] == condition)
                    & (variant_df["seed"] == seed)
                ].iloc[0]

                random_row = random_by_seed.loc[seed]

                trained_value = float(trained[METRIC])

                controls = {
                    "random_initialization": float(
                        random_row[METRIC]
                    ),
                    "spatial_shuffle": float(
                        shuffled[METRIC].mean()
                    ),
                    "pixel_permutation_model": float(
                        pixel_model[METRIC]
                    ),
                    "pixel_permutation_original_image": float(
                        pixel_image[METRIC]
                    ),
                }

                for comparison, control_value in controls.items():
                    rows.append(
                        {
                            "sensitivity_id": variant,
                            "condition": condition,
                            "metric": METRIC,
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
        len(VARIANTS)
        * len(CONDITIONS)
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
        [
            "sensitivity_id",
            "condition",
            "metric",
            "comparison",
        ],
        sort=False,
    )

    for (
        variant,
        condition,
        metric,
        comparison,
    ), group in grouped:
        group = group.sort_values("seed")

        require(
            group["seed"].tolist() == SEEDS,
            f"Incomplete seed pairing for "
            f"{variant}, {condition}, {comparison}.",
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
                "sensitivity_id": variant,
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
        OUTPUT_DIR
        / "structural_sensitivity_paired_seed_values.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "structural_sensitivity_paired_summary.csv",
        index=False,
    )

    print("INPUT VALIDATION: OK")
    print()

    print(
        summary[
            [
                "sensitivity_id",
                "condition",
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
