from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DATA_DIR = Path(
    "results/structural/spatial_template_reuse"
)
FIGURE_DIR = Path(
    "figures/spatial_template_reuse"
)
CHECKPOINT_DIR = Path(
    "results/final/training/checkpoints"
)

DATA_PATH = (
    DATA_DIR / "spatial_template_reuse_data.npz"
)
ASSIGNMENTS_PATH = (
    DATA_DIR / "spatial_template_reuse_assignments.csv"
)

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

LABELS = {
    "baseline": "Baseline",
    "exposure": "Exposure",
    "consistency": "Consistency",
    "augmentation": "Augmentation",
    "augmentation_consistency": "Aug. + Cons.",
}

PATCH_SIZE = 7
EXAMPLES = 7


def load_weights(condition, seed):
    path = (
        CHECKPOINT_DIR
        / f"mlp_{condition}_seed{seed}_best.pt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    weights = checkpoint[
        "model_state_dict"
    ]["input_projection.weight"]

    weights = (
        weights.detach()
        .cpu()
        .reshape(512, 3, 32, 32)
    )

    return weights


def energy_map(weight):
    energy = (
        weight.square()
        .sum(dim=0)
        .numpy()
    )

    maximum = energy.max()

    if maximum > 0:
        energy = energy / maximum

    return energy


def select_examples(
    rows,
):
    """
    Divide the valid horizontal image range into seven
    bins and take the strongest template match in each.

    This gives an automatic left-to-right visualization
    of reuse without manually choosing examples.
    """

    rows = rows.copy()

    minimum_center = PATCH_SIZE // 2
    maximum_center = (
        32 - PATCH_SIZE
        + PATCH_SIZE // 2
    )

    edges = np.linspace(
        minimum_center,
        maximum_center + 1,
        EXAMPLES + 1,
    )

    selected = []

    for index in range(EXAMPLES):
        lower = edges[index]
        upper = edges[index + 1]

        if index == EXAMPLES - 1:
            mask = (
                (rows["patch_center_column"] >= lower)
                & (rows["patch_center_column"] <= upper)
            )
        else:
            mask = (
                (rows["patch_center_column"] >= lower)
                & (rows["patch_center_column"] < upper)
            )

        candidates = rows[mask]

        if candidates.empty:
            selected.append(None)
            continue

        best_index = (
            candidates[
                "squared_template_match"
            ]
            .idxmax()
        )

        selected.append(
            candidates.loc[best_index]
        )

    return selected


def get_selected_examples(
    data,
    assignments,
):
    output = {}

    for condition in CONDITIONS:
        order = data[
            f"{condition}_template_order"
        ].astype(int)

        most_used_template = int(
            order[0]
        )

        rows = assignments[
            (
                assignments["condition"]
                == condition
            )
            & (
                assignments["template_id"]
                == most_used_template
            )
        ].copy()

        output[condition] = {
            "template_id": most_used_template,
            "examples": select_examples(rows),
        }

    return output


def plot_full_filters(
    selections,
):
    weight_cache = {}

    fig, axes = plt.subplots(
        len(CONDITIONS),
        EXAMPLES,
        figsize=(11.5, 7.6),
    )

    fig.suptitle(
        "First-layer filters sharing each condition-specific dominant local template",
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )

    for row_index, condition in enumerate(
        CONDITIONS
    ):
        examples = selections[
            condition
        ]["examples"]

        for column_index, example in enumerate(
            examples
        ):
            ax = axes[
                row_index,
                column_index,
            ]

            ax.set_xticks([])
            ax.set_yticks([])

            if example is None:
                ax.axis("off")
                continue

            seed = int(
                example["seed"]
            )

            neuron = int(
                example["neuron_index"]
            )

            if (condition, seed) not in weight_cache:
                weight_cache[
                    condition,
                    seed,
                ] = load_weights(
                    condition,
                    seed,
                )

            weight = weight_cache[
                condition,
                seed,
            ][neuron]

            image = energy_map(
                weight
            )

            ax.imshow(
                image,
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

            if row_index == 0:
                if column_index == 0:
                    title = "Left"
                elif column_index == 3:
                    title = "Centre"
                elif column_index == 6:
                    title = "Right"
                else:
                    title = ""

                ax.set_title(
                    title,
                    fontsize=9.5,
                    fontweight=(
                        "semibold"
                        if title
                        else "normal"
                    ),
                    pad=5,
                )

        axes[row_index, 0].text(
            -0.22,
            0.5,
            LABELS[condition],
            transform=axes[
                row_index,
                0,
            ].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
        )


    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.93,
        bottom=0.08,
        wspace=0.045,
        hspace=0.08,
    )

    output = (
        FIGURE_DIR
        / "figure_gmlp_style_spatial_filter_reuse"
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


def main():
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.load(
        DATA_PATH
    )

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH
    )

    selections = get_selected_examples(
        data,
        assignments,
    )

    plot_full_filters(
        selections,
    )

    print(
        "gMLP-style spatial reuse figure created."
    )


if __name__ == "__main__":
    main()
