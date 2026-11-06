from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


INPUT = Path(
    "results/structural/analysis/merged/"
    "structural_analysis_all.csv"
)

OUTPUT = Path("figures/structural_controls")

SEEDS = [43, 44, 45, 46, 47]

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

CONDITION_LABELS = [
    "Baseline",
    "Exposure",
    "Consistency",
    "Augmentation",
    "Aug. +\nConsistency",
]

CONTROLS = [
    (
        "random",
        "Random initialization",
    ),
    (
        "shuffle",
        "Spatial shuffle",
    ),
    (
        "pixel_model",
        "Pixel permutation\n(model coordinates)",
    ),
    (
        "pixel_original",
        "Pixel permutation\n(original-image coordinates)",
    ),
]

METRICS = [
    (
        "mean_locality_window_7",
        "Locality",
        "Absolute 7×7 locality score\n(higher = more structure)",
        "figure_structural_controls_absolute_locality",
    ),
    (
        "mean_template_total_reconstruction",
        "Template reuse",
        "Absolute template reconstruction fraction\n(higher = more structure)",
        "figure_structural_controls_absolute_template_reuse",
    ),
]

TRAINED_COLOR = "#1f4e79"
CONTROL_COLOR = "#b7b7b7"


def validate(data):
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

    if len(data) != 580:
        raise ValueError(
            f"Expected 580 rows, found {len(data)}"
        )


def values_for(
    data,
    metric,
    condition,
    source,
):
    if source == "trained":
        subset = data[
            (data["source"] == "trained")
            & (data["condition"] == condition)
            & (
                data["coordinate_system"]
                == "original_image"
            )
        ]

        values = (
            subset
            .set_index("seed")
            .loc[SEEDS, metric]
            .to_numpy(float)
        )

    elif source == "random":
        subset = data[
            data["source"]
            == "random_initialization"
        ]

        values = (
            subset
            .set_index("seed")
            .loc[SEEDS, metric]
            .to_numpy(float)
        )

    elif source == "shuffle":
        subset = data[
            (data["source"] == "spatial_shuffle")
            & (data["condition"] == condition)
        ]

        values = (
            subset
            .groupby("seed")[metric]
            .mean()
            .loc[SEEDS]
            .to_numpy(float)
        )

    elif source == "pixel_model":
        subset = data[
            (data["source"] == "pixel_permutation")
            & (data["condition"] == condition)
            & (
                data["coordinate_system"]
                == "model"
            )
        ]

        values = (
            subset
            .set_index("seed")
            .loc[SEEDS, metric]
            .to_numpy(float)
        )

    elif source == "pixel_original":
        subset = data[
            (data["source"] == "pixel_permutation")
            & (data["condition"] == condition)
            & (
                data["coordinate_system"]
                == "original_image"
            )
        ]

        values = (
            subset
            .set_index("seed")
            .loc[SEEDS, metric]
            .to_numpy(float)
        )

    else:
        raise ValueError(source)

    if len(values) != 5:
        raise ValueError(
            f"Expected 5 values for "
            f"{source}/{condition}/{metric}"
        )

    return values


def summarize(values):
    mean = float(values.mean())

    critical = t.ppf(
        0.975,
        df=len(values) - 1,
    )

    ci = float(
        critical
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )

    return mean, ci


def plot_metric(
    data,
    metric,
    metric_title,
    ylabel,
    filename,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 9.0),
        sharey=True,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.12,
        top=0.84,
        wspace=0.13,
        hspace=0.34,
    )

    x = np.arange(
        len(CONDITIONS)
    )

    width = 0.34

    all_upper_values = []

    stored = {}

    for control, control_title in CONTROLS:
        trained_means = []
        trained_errors = []

        control_means = []
        control_errors = []

        for condition in CONDITIONS:
            trained_values = values_for(
                data,
                metric,
                condition,
                "trained",
            )

            control_values = values_for(
                data,
                metric,
                condition,
                control,
            )

            trained_mean, trained_ci = summarize(
                trained_values
            )

            control_mean, control_ci = summarize(
                control_values
            )

            trained_means.append(
                trained_mean
            )

            trained_errors.append(
                trained_ci
            )

            control_means.append(
                control_mean
            )

            control_errors.append(
                control_ci
            )

            all_upper_values.extend(
                [
                    trained_mean + trained_ci,
                    control_mean + control_ci,
                ]
            )

        stored[control] = {
            "title": control_title,
            "trained_mean": np.asarray(
                trained_means
            ),
            "trained_error": np.asarray(
                trained_errors
            ),
            "control_mean": np.asarray(
                control_means
            ),
            "control_error": np.asarray(
                control_errors
            ),
        }

    ymax = max(
        all_upper_values
    ) * 1.10

    for ax, (
        control,
        control_title,
    ) in zip(
        axes.flat,
        CONTROLS,
    ):
        result = stored[control]

        ax.bar(
            x - width / 2,
            result["trained_mean"],
            width=width,
            yerr=result["trained_error"],
            color=TRAINED_COLOR,
            edgecolor="white",
            linewidth=0.8,
            error_kw={
                "elinewidth": 1.0,
                "capsize": 3.0,
                "ecolor": "0.25",
            },
            label="Trained",
            zorder=3,
        )

        ax.bar(
            x + width / 2,
            result["control_mean"],
            width=width,
            yerr=result["control_error"],
            color=CONTROL_COLOR,
            edgecolor="white",
            linewidth=0.8,
            error_kw={
                "elinewidth": 1.0,
                "capsize": 3.0,
                "ecolor": "0.25",
            },
            label="Control",
            zorder=3,
        )

        ax.set_ylim(
            0,
            ymax,
        )

        ax.set_title(
            control_title,
            fontsize=12.5,
            fontweight="semibold",
            pad=13,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            CONDITION_LABELS,
            fontsize=10,
        )

        ax.tick_params(
            axis="y",
            labelsize=9.5,
        )

        ax.grid(
            axis="y",
            alpha=0.15,
            linewidth=0.8,
            zorder=0,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_alpha(0.45)
        ax.spines["bottom"].set_alpha(0.45)

    axes[0, 0].set_ylabel(
        ylabel,
        fontsize=10.5,
    )

    axes[1, 0].set_ylabel(
        ylabel,
        fontsize=10.5,
    )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=TRAINED_COLOR,
            label="Trained MLP",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=CONTROL_COLOR,
            label="Control",
        ),
    ]

    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.025),
        ncol=2,
        columnspacing=2.2,
    )

    fig.suptitle(
        f"{metric_title}: trained MLP versus structural controls",
        fontsize=14,
        fontweight="semibold",
        y=0.95,
    )

    png = OUTPUT / f"{filename}.png"
    pdf = OUTPUT / f"{filename}.pdf"

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

    print(f"Saved: {png}")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    data = pd.read_csv(INPUT)

    validate(data)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        metric,
        metric_title,
        ylabel,
        filename,
    ) in METRICS:
        plot_metric(
            data,
            metric,
            metric_title,
            ylabel,
            filename,
        )

    print("DONE")


if __name__ == "__main__":
    main()
