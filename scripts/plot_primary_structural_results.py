from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


INPUT = Path(
    "results/structural/analysis/merged/"
    "structural_analysis_all.csv"
)

OUTPUT = Path("figures/structural_results")

SEEDS = [43, 44, 45, 46, 47]

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

LABELS = [
    "Baseline",
    "Exposure",
    "Consistency",
    "Augmentation",
    "Aug. +\nConsistency",
]

METRICS = [
    (
        "mean_locality_window_7",
        "Locality",
        "Maximum 7×7 window energy fraction",
    ),
    (
        "mean_template_total_reconstruction",
        "Template reuse",
        "Template reconstruction fraction",
    ),
]


def validate(data):
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    required = {
        "source",
        "coordinate_system",
        "condition",
        "seed",
        "mean_locality_window_7",
        "mean_template_total_reconstruction",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )


def trained_table(data):
    result = data[
        (data["source"] == "trained")
        & (
            data["coordinate_system"]
            == "original_image"
        )
    ].copy()

    expected = len(CONDITIONS) * len(SEEDS)

    if len(result) != expected:
        raise ValueError(
            f"Expected {expected} trained rows, "
            f"found {len(result)}"
        )

    for condition in CONDITIONS:
        subset = result[
            result["condition"] == condition
        ]

        if sorted(
            subset["seed"].astype(int).tolist()
        ) != SEEDS:
            raise ValueError(
                f"Invalid seeds for {condition}"
            )

    return result


def summarize(data, metric):
    critical = t.ppf(
        0.975,
        df=len(SEEDS) - 1,
    )

    rows = []

    for condition in CONDITIONS:
        subset = (
            data[
                data["condition"] == condition
            ]
            .sort_values("seed")
        )

        values = subset[metric].to_numpy(
            dtype=float
        )

        mean = values.mean()
        sd = values.std(ddof=1)

        half_width = (
            critical
            * sd
            / np.sqrt(len(values))
        )

        rows.append(
            {
                "condition": condition,
                "mean": mean,
                "lower": mean - half_width,
                "upper": mean + half_width,
            }
        )

    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(INPUT)

    validate(data)

    trained = trained_table(data)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.9),
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.18,
        top=0.78,
        wspace=0.16,
    )

    x = np.arange(len(CONDITIONS))

    for ax, (
        metric,
        title,
        ylabel,
    ) in zip(
        axes,
        METRICS,
    ):
        summary = summarize(
            trained,
            metric,
        )

        for index, condition in enumerate(
            CONDITIONS
        ):
            subset = (
                trained[
                    trained["condition"]
                    == condition
                ]
                .sort_values("seed")
            )

            values = subset[
                metric
            ].to_numpy(dtype=float)

            jitter = np.linspace(
                -0.055,
                0.055,
                len(values),
            )

            ax.scatter(
                np.full(
                    len(values),
                    index,
                    dtype=float,
                )
                + jitter,
                values,
                s=30,
                alpha=0.32,
                color="0.45",
                edgecolors="none",
                zorder=2,
            )

        means = summary[
            "mean"
        ].to_numpy()

        lower_error = (
            means
            - summary["lower"].to_numpy()
        )

        upper_error = (
            summary["upper"].to_numpy()
            - means
        )

        ax.errorbar(
            x,
            means,
            yerr=[
                lower_error,
                upper_error,
            ],
            fmt="o",
            markersize=7,
            linewidth=1.8,
            capsize=4,
            color="#1f4e79",
            ecolor="#1f4e79",
            markerfacecolor="#1f4e79",
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )

        ax.set_title(
            title,
            fontsize=12.5,
            fontweight="semibold",
            pad=12,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=10,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            LABELS,
            fontsize=9,
        )

        ax.tick_params(
            axis="y",
            labelsize=9,
        )

        ax.grid(
            axis="y",
            alpha=0.16,
            linewidth=0.8,
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.spines[
            "left"
        ].set_alpha(0.5)

        ax.spines[
            "bottom"
        ].set_alpha(0.5)

    fig.suptitle(
        "First-layer structure across MLP training conditions",
        fontsize=14,
        fontweight="semibold",
        y=0.94,
    )

    png = (
        OUTPUT
        / "figure_primary_structural_results.png"
    )

    pdf = (
        OUTPUT
        / "figure_primary_structural_results.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("DONE")
    print(f"Figure: {png}")


if __name__ == "__main__":
    main()
