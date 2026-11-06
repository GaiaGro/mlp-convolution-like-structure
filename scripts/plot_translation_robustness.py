from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


INPUT = Path(
    "results/final/evaluation/test_runs.csv"
)

OUTPUT = Path(
    "figures/performance"
)

SEEDS = [43, 44, 45, 46, 47]

SERIES = [
    ("mlp", "baseline", "MLP baseline"),
    ("mlp", "exposure", "MLP exposure"),
    ("mlp", "consistency", "MLP consistency"),
    ("mlp", "augmentation", "MLP augmentation"),
    (
        "mlp",
        "augmentation_consistency",
        "MLP aug. + consistency",
    ),
    ("cnn", "baseline", "CNN baseline"),
    ("cnn", "augmentation", "CNN augmentation"),
]

DISTANCES = [0, 1, 2, 4]

DISTANCE_COLUMNS = {
    0: "clean_accuracy",
    1: "distance1_mean_accuracy",
    2: "distance2_mean_accuracy",
    4: "distance4_mean_accuracy",
}


def validate(data):
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    required = {
        "model",
        "condition",
        "seed",
        *DISTANCE_COLUMNS.values(),
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(data) != 35:
        raise ValueError(
            f"Expected 35 runs, found {len(data)}"
        )


def summarize_series(data, model, condition):
    subset = data[
        (data["model"] == model)
        & (data["condition"] == condition)
    ].copy()

    if sorted(
        subset["seed"].astype(int).tolist()
    ) != SEEDS:
        raise ValueError(
            f"Invalid seeds for {model}/{condition}"
        )

    subset = subset.sort_values("seed")

    critical = t.ppf(
        0.975,
        df=len(SEEDS) - 1,
    )

    rows = []

    for distance in DISTANCES:
        column = DISTANCE_COLUMNS[
            distance
        ]

        values = subset[
            column
        ].to_numpy(dtype=float)

        mean = values.mean()
        sd = values.std(ddof=1)

        half_width = (
            critical
            * sd
            / np.sqrt(len(values))
        )

        rows.append(
            {
                "distance": distance,
                "mean": mean,
                "lower": mean - half_width,
                "upper": mean + half_width,
            }
        )

    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(INPUT)
    validate(data)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.9, 5.1),
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.78,
        bottom=0.14,
        top=0.86,
    )

    mlp_styles = [
        ("#b7c9d6", "o"),
        ("#8faec2", "o"),
        ("#668fa9", "o"),
        ("#3f7395", "o"),
        ("#1f4e79", "o"),
    ]

    cnn_styles = [
        ("#8a6d3b", "s"),
        ("#b9822f", "s"),
    ]

    mlp_index = 0
    cnn_index = 0

    for model, condition, label in SERIES:
        summary = summarize_series(
            data,
            model,
            condition,
        )

        x = summary[
            "distance"
        ].to_numpy(dtype=float)

        y = 100 * summary[
            "mean"
        ].to_numpy(dtype=float)

        lower = 100 * summary[
            "lower"
        ].to_numpy(dtype=float)

        upper = 100 * summary[
            "upper"
        ].to_numpy(dtype=float)

        if model == "mlp":
            color, marker = (
                mlp_styles[mlp_index]
            )
            mlp_index += 1
            linestyle = "-"
            linewidth = (
                2.1
                if condition
                == "augmentation_consistency"
                else 1.45
            )
        else:
            color, marker = (
                cnn_styles[cnn_index]
            )
            cnn_index += 1
            linestyle = "--"
            linewidth = 1.65

        ax.plot(
            x,
            y,
            marker=marker,
            markersize=5.5,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label,
            zorder=3,
        )

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )

    ax.set_title(
        "Translation robustness on CIFAR-10",
        fontsize=14,
        fontweight="semibold",
        pad=12,
    )

    ax.set_xlabel(
        "Translation distance (pixels)",
        fontsize=10,
    )

    ax.set_ylabel(
        "Test accuracy (%)",
        fontsize=10,
    )

    ax.set_xticks(
        DISTANCES
    )

    ax.tick_params(
        labelsize=9.5,
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

    ax.legend(
        frameon=False,
        fontsize=8.4,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        handlelength=2.4,
        labelspacing=0.7,
    )



    png = (
        OUTPUT
        / "figure_translation_robustness.png"
    )

    pdf = (
        OUTPUT
        / "figure_translation_robustness.pdf"
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
