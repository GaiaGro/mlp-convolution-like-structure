from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_PATH = Path(
    "results/structural/analysis/merged/"
    "structural_analysis_all.csv"
)

SENSITIVITY_PATH = Path(
    "results/structural/sensitivity/"
    "structural_sensitivity_all.csv"
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

CONDITION_LABELS = [
    "Baseline",
    "Exposure",
    "Consistency",
    "Augmentation",
    "Aug. +\nConsistency",
]

METRIC = "mean_template_total_reconstruction"


def validate(data, sensitivity=False):
    required = {
        "source",
        "coordinate_system",
        "condition",
        "seed",
        METRIC,
    }

    if sensitivity:
        required.add("sensitivity_id")

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )


def trained_minus_shuffle(
    data,
    sensitivity_id=None,
):
    if sensitivity_id is not None:
        data = data[
            data["sensitivity_id"] == sensitivity_id
        ].copy()

    values = []

    for condition in CONDITIONS:
        seed_differences = []

        for seed in SEEDS:
            subset = data[
                (data["condition"] == condition)
                & (data["seed"].astype(int) == seed)
            ]

            trained = subset[
                (subset["source"] == "trained")
                & (
                    subset["coordinate_system"]
                    == "original_image"
                )
            ]

            shuffled = subset[
                subset["source"]
                == "spatial_shuffle"
            ]

            if len(trained) != 1:
                raise ValueError(
                    f"Expected 1 trained row for "
                    f"{condition}, seed {seed}; "
                    f"found {len(trained)}"
                )

            if len(shuffled) != 20:
                raise ValueError(
                    f"Expected 20 shuffle rows for "
                    f"{condition}, seed {seed}; "
                    f"found {len(shuffled)}"
                )

            trained_value = float(
                trained.iloc[0][METRIC]
            )

            shuffle_value = float(
                shuffled[METRIC].mean()
            )

            seed_differences.append(
                trained_value - shuffle_value
            )

        values.append(
            float(
                np.mean(seed_differences)
            )
        )

    return np.asarray(values)


def load_values():
    if not PRIMARY_PATH.exists():
        raise FileNotFoundError(PRIMARY_PATH)

    if not SENSITIVITY_PATH.exists():
        raise FileNotFoundError(
            SENSITIVITY_PATH
        )

    primary = pd.read_csv(
        PRIMARY_PATH
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_PATH
    )

    validate(
        primary,
        sensitivity=False,
    )

    validate(
        sensitivity,
        sensitivity=True,
    )

    expected = {
        "patch5_templates16",
        "patch9_templates16",
        "patch7_templates8",
        "patch7_templates32",
    }

    observed = set(
        sensitivity[
            "sensitivity_id"
        ].unique()
    )

    if observed != expected:
        raise ValueError(
            "Unexpected sensitivity variants: "
            f"{sorted(observed)}"
        )

    primary_values = trained_minus_shuffle(
        primary
    )

    patch_matrix = np.column_stack(
        [
            trained_minus_shuffle(
                sensitivity,
                "patch5_templates16",
            ),
            primary_values,
            trained_minus_shuffle(
                sensitivity,
                "patch9_templates16",
            ),
        ]
    )

    template_matrix = np.column_stack(
        [
            trained_minus_shuffle(
                sensitivity,
                "patch7_templates8",
            ),
            primary_values,
            trained_minus_shuffle(
                sensitivity,
                "patch7_templates32",
            ),
        ]
    )

    return patch_matrix, template_matrix


def annotate_heatmap(
    ax,
    matrix,
):
    threshold = (
        matrix.min()
        + 0.55
        * (
            matrix.max()
            - matrix.min()
        )
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            value = matrix[
                row,
                column,
            ]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight=(
                    "semibold"
                    if column == 1
                    else "normal"
                ),
            )


def draw_panel(
    ax,
    matrix,
    column_labels,
    title,
    vmin,
    vmax,
):
    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(
        title,
        fontsize=12.5,
        fontweight="semibold",
        pad=12,
    )

    ax.set_xticks(
        np.arange(
            len(column_labels)
        )
    )

    ax.set_xticklabels(
        column_labels,
        fontsize=10,
    )

    ax.set_yticks(
        np.arange(
            len(CONDITION_LABELS)
        )
    )

    ax.set_yticklabels(
        CONDITION_LABELS,
        fontsize=10,
    )

    ax.tick_params(
        length=0,
    )

    # Thin white separators between cells.
    ax.set_xticks(
        np.arange(-0.5, 3, 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, 5, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=2.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Highlight the primary configuration:
    # the middle column.
    ax.add_patch(
        plt.Rectangle(
            (0.5, -0.5),
            1,
            len(CONDITIONS),
            fill=False,
            edgecolor="#163a5f",
            linewidth=1.8,
        )
    )

    annotate_heatmap(
        ax,
        matrix,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def main():
    (
        patch_matrix,
        template_matrix,
    ) = load_values()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_max = max(
        patch_matrix.max(),
        template_matrix.max(),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.15),
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.89,
        bottom=0.18,
        top=0.79,
        wspace=0.42,
    )

    image = draw_panel(
        axes[0],
        patch_matrix,
        [
            "5×5",
            "7×7\n(primary)",
            "9×9",
        ],
        "Patch-size sensitivity",
        vmin=0,
        vmax=global_max,
    )

    draw_panel(
        axes[1],
        template_matrix,
        [
            "8",
            "16\n(primary)",
            "32",
        ],
        "Template-count sensitivity",
        vmin=0,
        vmax=global_max,
    )

    axes[1].set_yticklabels([])

    fig.suptitle(
        "Robustness of learned template reuse",
        fontsize=14,
        fontweight="semibold",
        y=0.94,
    )

    cbar_ax = fig.add_axes(
        [
            0.91,
            0.22,
            0.014,
            0.48,
        ]
    )

    cbar = fig.colorbar(
        image,
        cax=cbar_ax,
    )

    cbar.set_label(
        "Trained − shuffle effect",
        fontsize=9,
    )

    cbar.ax.tick_params(
        labelsize=8.5,
    )



    png = (
        OUTPUT
        / "figure_structural_sensitivity.png"
    )

    pdf = (
        OUTPUT
        / "figure_structural_sensitivity.pdf"
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
