from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTION_DIR = Path(
    "results/final/evaluation/predictions"
)

OUTPUT_DIR = Path(
    "figures/performance"
)

STATISTICS_DIR = Path(
    "results/final/evaluation/statistics"
)

SEEDS = [43, 44, 45, 46, 47]

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PANELS = [
    ("mlp", "baseline", "MLP — Baseline"),
    (
        "mlp",
        "augmentation_consistency",
        "MLP — Aug. + Consistency",
    ),
    ("cnn", "baseline", "CNN — Baseline"),
    ("cnn", "augmentation", "CNN — Augmentation"),
]


def confusion_matrix(targets, predictions):
    matrix = np.zeros(
        (len(CLASSES), len(CLASSES)),
        dtype=np.int64,
    )

    np.add.at(
        matrix,
        (targets, predictions),
        1,
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return matrix / row_totals * 100.0


def load_seed_matrix(
    model,
    condition,
    seed,
):
    path = (
        PREDICTION_DIR
        / f"{model}_{condition}_seed{seed}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        targets = data[
            "targets"
        ].astype(int)

        predictions = data[
            "clean_predictions"
        ].astype(int)

    return confusion_matrix(
        targets,
        predictions,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATISTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_results = {}
    summary_rows = []

    for model, condition, _title in PANELS:
        matrices = [
            load_seed_matrix(
                model,
                condition,
                seed,
            )
            for seed in SEEDS
        ]

        stack = np.stack(
            matrices,
            axis=0,
        )

        mean_matrix = stack.mean(
            axis=0
        )

        std_matrix = stack.std(
            axis=0,
            ddof=1,
        )

        panel_results[
            (model, condition)
        ] = mean_matrix

        for i, true_class in enumerate(CLASSES):
            for j, predicted_class in enumerate(CLASSES):
                summary_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "mean_percent": mean_matrix[i, j],
                        "std_percent": std_matrix[i, j],
                        "n_seeds": len(SEEDS),
                    }
                )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        STATISTICS_DIR
        / "confusion_matrix_summary.csv",
        index=False,
    )

    # Separate column reserved exclusively
    # for the shared colour bar.
    fig = plt.figure(
        figsize=(14.5, 11.5)
    )

    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[
            1.0,
            1.0,
            0.045,
        ],
        left=0.08,
        right=0.93,
        bottom=0.09,
        top=0.91,
        wspace=0.30,
        hspace=0.34,
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    cax = fig.add_subplot(
        grid[:, 2]
    )

    image = None

    for ax, (
        model,
        condition,
        title,
    ) in zip(
        axes,
        PANELS,
    ):
        matrix = panel_results[
            (model, condition)
        ]

        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=100,
            interpolation="nearest",
        )

        ax.set_title(
            title,
            fontsize=12.5,
            fontweight="semibold",
            pad=10,
        )

        ax.set_xticks(
            np.arange(len(CLASSES))
        )

        ax.set_yticks(
            np.arange(len(CLASSES))
        )

        ax.set_xticklabels(
            CLASSES,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=8.2,
        )

        ax.set_yticklabels(
            CLASSES,
            fontsize=8.2,
        )

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                value = matrix[i, j]

                text_color = (
                    "white"
                    if value >= 50
                    else "#303030"
                )

                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=text_color,
                )

    colorbar = fig.colorbar(
        image,
        cax=cax,
    )

    colorbar.set_label(
        "Predictions (%)",
        fontsize=10,
        labelpad=9,
    )

    colorbar.ax.tick_params(
        labelsize=8.5,
    )

    fig.suptitle(
        "CIFAR-10 classification patterns across models",
        fontsize=14,
        fontweight="semibold",
        y=0.965,
    )

    for ax in axes:
        ax.set_xlabel(
            "Predicted class",
            fontsize=9.5,
            labelpad=-2,
        )

        ax.set_ylabel(
            "True class",
            fontsize=9.5,
            labelpad=-3,
        )

    output = (
        OUTPUT_DIR
        / "figure_confusion_matrices"
    )

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

    print("Confusion-matrix figure created.")


if __name__ == "__main__":
    main()
