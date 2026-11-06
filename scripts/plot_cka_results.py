from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results/cka")
FIGURE_DIR = Path("figures/cka")
SUMMARY_PATH = RESULTS_DIR / "cka_paired_summary.csv"

MLP_CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

CONDITION_LABELS = {
    "baseline": "Baseline MLP",
    "exposure": "Exposure MLP",
    "consistency": "Consistency MLP",
    "augmentation": "Augmentation MLP",
    "augmentation_consistency": "Aug. + Consistency MLP",
}

MLP_LAYERS = [
    "Projection",
    "Block 1",
    "Block 2",
    "Block 3",
    "Output norm",
]

CNN_LAYERS = [
    "Stem",
    "Stage 1",
    "Stage 2",
    "Stage 3a",
    "Stage 3b",
    "Stage 4",
    "Global pool",
]


def build_matrix(summary, cnn_condition, mlp_condition):
    subset = summary[
        (summary["cnn_condition"] == cnn_condition)
        & (summary["mlp_condition"] == mlp_condition)
    ]

    matrix = np.empty(
        (len(MLP_LAYERS), len(CNN_LAYERS)),
        dtype=float,
    )

    for i, mlp_layer in enumerate(MLP_LAYERS):
        for j, cnn_layer in enumerate(CNN_LAYERS):
            cell = subset[
                (subset["mlp_layer"] == mlp_layer)
                & (subset["cnn_layer"] == cnn_layer)
            ]

            if len(cell) != 1:
                raise RuntimeError(
                    "Missing or duplicated CKA summary cell."
                )

            matrix[i, j] = float(
                cell.iloc[0]["mean_cka"]
            )

    return matrix


def plot_reference(
    summary,
    cnn_condition,
    output_name,
    figure_title,
    vmin,
    vmax,
):
    fig = plt.figure(figsize=(14.8, 8.8))

    grid = fig.add_gridspec(
        2,
        7,
        width_ratios=[
            1, 1,
            1, 1,
            1, 1,
            0.10,
        ],
        left=0.08,
        right=0.93,
        bottom=0.13,
        top=0.89,
        wspace=0.95,
        hspace=0.55,
    )

    positions = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (1, slice(1, 3)),
        (1, slice(3, 5)),
    ]

    axes = []
    image = None

    for condition, position in zip(
        MLP_CONDITIONS,
        positions,
    ):
        row, columns = position
        ax = fig.add_subplot(grid[row, columns])
        axes.append(ax)

        matrix = build_matrix(
            summary,
            cnn_condition,
            condition,
        )

        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(
            CONDITION_LABELS[condition],
            fontsize=11.5,
            fontweight="semibold",
            pad=9,
        )

        ax.set_xticks(
            np.arange(len(CNN_LAYERS))
        )
        ax.set_xticklabels(
            CNN_LAYERS,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=8.2,
        )

        ax.set_yticks(
            np.arange(len(MLP_LAYERS))
        )
        ax.set_yticklabels(
            MLP_LAYERS,
            fontsize=8.5,
        )

        ax.tick_params(
            axis="both",
            length=3,
        )

    cax = fig.add_subplot(grid[:, 6])

    colorbar = fig.colorbar(
        image,
        cax=cax,
    )

    colorbar.set_label(
        "Linear CKA",
        fontsize=9.5,
        labelpad=8,
    )

    colorbar.ax.tick_params(
        labelsize=8.5,
    )

    fig.suptitle(
        figure_title,
        fontsize=14,
        fontweight="semibold",
        y=0.965,
    )

    fig.text(
        0.50,
        0.045,
        "CNN layer",
        ha="center",
        fontsize=10.5,
    )

    fig.text(
        0.025,
        0.50,
        "MLP layer",
        va="center",
        rotation="vertical",
        fontsize=10.5,
    )

    output = FIGURE_DIR / output_name

    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(SUMMARY_PATH)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    # One common colour scale for both figures.
    # Limits are derived from all observed mean CKA
    # values and rounded outward to 0.05.
    observed_min = float(summary["mean_cka"].min())
    observed_max = float(summary["mean_cka"].max())

    step = 0.05

    vmin = max(
        0.0,
        np.floor(observed_min / step) * step,
    )
    vmax = min(
        1.0,
        np.ceil(observed_max / step) * step,
    )

    print(
        f"Shared CKA colour scale: "
        f"{vmin:.2f} to {vmax:.2f}"
    )

    plot_reference(
        summary,
        cnn_condition="baseline",
        output_name=(
            "figure_cka_mlp_vs_cnn_baseline"
        ),
        figure_title=(
            "Representational similarity "
            "to the baseline CNN"
        ),
        vmin=vmin,
        vmax=vmax,
    )

    plot_reference(
        summary,
        cnn_condition="augmentation",
        output_name=(
            "figure_cka_mlp_vs_cnn_augmentation_sensitivity"
        ),
        figure_title=(
            "Representational similarity "
            "to the augmentation-trained CNN"
        ),
        vmin=vmin,
        vmax=vmax,
    )

    print("CKA figures created.")


if __name__ == "__main__":
    main()
