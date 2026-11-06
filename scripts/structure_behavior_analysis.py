from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [43, 44, 45, 46, 47]

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

STRUCTURE = {
    "mean_locality_window_7": "Locality (7 x 7)",
    "mean_template_total_reconstruction": (
        "Template-reuse reconstruction"
    ),
}

BEHAVIOR = {
    "mean_shifted_accuracy": "Mean shifted accuracy",
    "shifted_retention": "Mean shifted / clean accuracy",
    "heldout_distance4_mean_accuracy": (
        "Held-out distance-4 accuracy"
    ),
    "distance4_retention": "Distance-4 / clean accuracy",
    "mean_prediction_consistency": (
        "Mean prediction consistency"
    ),
}

COLORS = {
    "baseline": "#b7c9d6",
    "exposure": "#8faec2",
    "consistency": "#668fa9",
    "augmentation": "#3f7395",
    "augmentation_consistency": "#1f4e79",
}

LABELS = {
    "baseline": "Baseline",
    "exposure": "Exposure",
    "consistency": "Consistency",
    "augmentation": "Augmentation",
    "augmentation_consistency": "Aug. + consistency",
}

OUTPUT = Path(
    "results/structural/structure_behavior"
)


def correlation(first, second):
    first = np.asarray(
        first,
        dtype=float,
    ).ravel()

    second = np.asarray(
        second,
        dtype=float,
    ).ravel()

    if (
        np.std(first) == 0
        or np.std(second) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(first, second)[0, 1]
    )


def center(matrix, axis):
    return (
        matrix
        - matrix.mean(
            axis=axis,
            keepdims=True,
        )
    )


def holm_adjust(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="stable",
    )

    ordered = values[order]

    adjusted_ordered = (
        np.maximum.accumulate(
            (
                len(values)
                - np.arange(len(values))
            )
            * ordered
        ).clip(max=1.0)
    )

    adjusted = np.empty_like(
        adjusted_ordered
    )

    adjusted[order] = adjusted_ordered
    return adjusted


def metric_matrix(table, metric):
    matrix = (
        table.pivot(
            index="seed",
            columns="condition",
            values=metric,
        )
        .loc[SEEDS, CONDITIONS]
        .to_numpy(float)
    )

    if (
        matrix.shape != (5, 5)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(
            f"Invalid matrix for {metric}."
        )

    return matrix


def association(
    first,
    second,
    adjustment,
    seed,
):
    generator = np.random.default_rng(
        seed
    )

    axis = (
        1
        if adjustment == "seed_adjusted"
        else 0
    )

    first_centered = center(
        first,
        axis,
    )

    second_centered = center(
        second,
        axis,
    )

    observed = correlation(
        first_centered,
        second_centered,
    )

    bootstrap = []

    for _ in range(10000):
        indices = generator.integers(
            0,
            5,
            size=5,
        )

        first_sample = first[indices]
        second_sample = second[indices]

        bootstrap.append(
            correlation(
                center(
                    first_sample,
                    axis,
                ),
                center(
                    second_sample,
                    axis,
                ),
            )
        )

    bootstrap = np.asarray(bootstrap)
    bootstrap = bootstrap[
        np.isfinite(bootstrap)
    ]

    ci_lower, ci_upper = np.quantile(
        bootstrap,
        [0.025, 0.975],
    )

    exceedances = 0

    for _ in range(20000):
        permuted = np.empty_like(
            second
        )

        if adjustment == "seed_adjusted":
            for row in range(5):
                permuted[row] = (
                    generator.permutation(
                        second[row]
                    )
                )
        else:
            for column in range(5):
                permuted[:, column] = (
                    generator.permutation(
                        second[:, column]
                    )
                )

        permuted = center(
            permuted,
            axis,
        )

        permuted_correlation = (
            correlation(
                first_centered,
                permuted,
            )
        )

        exceedances += (
            abs(permuted_correlation)
            >= abs(observed)
        )

    p_value = (
        (exceedances + 1)
        / 20001
    )

    return (
        observed,
        ci_lower,
        ci_upper,
        p_value,
    )


def load_table():
    structural = pd.read_csv(
        "results/structural/analysis/"
        "merged/structural_analysis_all.csv"
    )

    test = pd.read_csv(
        "results/final/evaluation/"
        "test_runs.csv"
    )

    structural = structural[
        (
            structural["source"]
            == "trained"
        )
        & (
            structural[
                "coordinate_system"
            ]
            == "original_image"
        )
    ]

    test = test[
        test["model"] == "mlp"
    ]

    if (
        len(structural) != 25
        or len(test) != 25
    ):
        raise ValueError(
            "Expected 25 trained MLP "
            "rows in both inputs."
        )

    table = structural[
        [
            "run_name",
            "condition",
            "seed",
            *STRUCTURE,
        ]
    ].merge(
        test[
            [
                "run_name",
                "condition",
                "seed",
                "clean_accuracy",
                "mean_shifted_accuracy",
                (
                    "heldout_distance4_"
                    "mean_accuracy"
                ),
                (
                    "mean_prediction_"
                    "consistency"
                ),
            ]
        ],
        on=[
            "run_name",
            "condition",
            "seed",
        ],
        validate="one_to_one",
    )

    if len(table) != 25:
        raise ValueError(
            "Structural and behavioral "
            "runs do not match."
        )

    if table.duplicated(
        ["seed", "condition"]
    ).any():
        raise ValueError(
            "Duplicate seed-condition pairs."
        )

    if set(table["seed"]) != set(SEEDS):
        raise ValueError(
            "Unexpected seeds."
        )

    if (
        set(table["condition"])
        != set(CONDITIONS)
    ):
        raise ValueError(
            "Unexpected conditions."
        )

    table["shifted_retention"] = (
        table["mean_shifted_accuracy"]
        / table["clean_accuracy"]
    )

    table["distance4_retention"] = (
        table[
            "heldout_distance4_mean_accuracy"
        ]
        / table["clean_accuracy"]
    )

    return (
        table.sort_values(
            ["seed", "condition"]
        )
        .reset_index(drop=True)
    )


def make_figure(table):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.2),
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.22,
        top=0.84,
        wspace=0.26,
    )

    behavior = "distance4_retention"

    for axis, (
        metric,
        x_label,
    ) in zip(
        axes,
        STRUCTURE.items(),
    ):
        for condition in CONDITIONS:
            subset = table[
                table["condition"]
                == condition
            ]

            x_values = subset[metric]
            y_values = 100 * subset[behavior]

            x_mean = x_values.mean()
            y_mean = y_values.mean()

            axis.scatter(
                x_values,
                y_values,
                color=COLORS[condition],
                s=34,
                alpha=0.38,
                edgecolor="none",
            )

            axis.errorbar(
                x_mean,
                y_mean,
                xerr=x_values.std(ddof=1),
                yerr=y_values.std(ddof=1),
                color=COLORS[condition],
                marker="o",
                markersize=7.5,
                markeredgecolor="black",
                markeredgewidth=0.55,
                capsize=3,
                linewidth=1.1,
                zorder=3,
            )

            axis.annotate(
                LABELS[condition],
                (x_mean, y_mean),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8.2,
            )

        axis.set_xlabel(x_label)

        axis.set_ylabel(
            "Held-out distance-4 accuracy "
            "retention (%)"
        )

        axis.grid(
            alpha=0.2,
            linewidth=0.7,
        )

    figure.suptitle(
        "Learned structure accompanies "
        "held-out translation robustness",
        fontsize=14,
        fontweight="semibold",
    )


    figure_directory = Path("figures/structural_results")
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figure_directory
        / "figure_structure_behavior_associations.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        figure_directory
        / "figure_structure_behavior_associations.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = load_table()

    table.to_csv(
        OUTPUT
        / "structure_behavior_models.csv",
        index=False,
    )

    metrics = [
        *STRUCTURE,
        "clean_accuracy",
        *BEHAVIOR,
    ]

    summary = (
        table.groupby("condition")[metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT
        / (
            "structure_behavior_"
            "condition_summary.csv"
        )
    )

    rows = []
    counter = 0

    for adjustment in [
        "seed_adjusted",
        "condition_adjusted",
    ]:
        for structure_metric in STRUCTURE:
            first = metric_matrix(
                table,
                structure_metric,
            )

            for behavior_metric in BEHAVIOR:
                second = metric_matrix(
                    table,
                    behavior_metric,
                )

                (
                    estimate,
                    lower,
                    upper,
                    p_value,
                ) = association(
                    first,
                    second,
                    adjustment,
                    2026 + counter,
                )

                counter += 1

                rows.append(
                    {
                        "analysis": adjustment,
                        "structure_metric": (
                            structure_metric
                        ),
                        "behavior_metric": (
                            behavior_metric
                        ),
                        "n_seeds": 5,
                        "n_models": 25,
                        "correlation": estimate,
                        (
                            "bootstrap_ci95_lower"
                        ): lower,
                        (
                            "bootstrap_ci95_upper"
                        ): upper,
                        (
                            "permutation_p_value"
                        ): p_value,
                    }
                )

    associations = pd.DataFrame(rows)

    associations[
        "holm_adjusted_p_value"
    ] = np.nan

    for adjustment in associations[
        "analysis"
    ].unique():
        selected = (
            associations["analysis"]
            == adjustment
        )

        associations.loc[
            selected,
            "holm_adjusted_p_value",
        ] = holm_adjust(
            associations.loc[
                selected,
                "permutation_p_value",
            ]
        )

    associations.to_csv(
        OUTPUT
        / "structure_behavior_associations.csv",
        index=False,
    )

    make_figure(table)

    primary = associations[
        associations["analysis"]
        == "seed_adjusted"
    ]

    print(
        primary[
            [
                "structure_metric",
                "behavior_metric",
                "correlation",
                "bootstrap_ci95_lower",
                "bootstrap_ci95_upper",
                "holm_adjusted_p_value",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print(
        f"\nResults saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
