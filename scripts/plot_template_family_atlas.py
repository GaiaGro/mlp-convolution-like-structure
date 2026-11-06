from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch


DATA_DIR = Path("results/structural/spatial_template_reuse")
CHECKPOINT_DIR = Path("results/final/training/checkpoints")
FIGURE_DIR = Path("figures/spatial_template_reuse")

DATA_PATH = DATA_DIR / "spatial_template_reuse_data.npz"
ASSIGNMENTS_PATH = DATA_DIR / "spatial_template_reuse_assignments.csv"

CONDITION = "augmentation_consistency"

SEEDS = [43, 44, 45, 46, 47]
PATCH_SIZE = 7


def load_weights(seed):
    path = (
        CHECKPOINT_DIR
        / f"mlp_{CONDITION}_seed{seed}_best.pt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    weights = (
        checkpoint["model_state_dict"][
            "input_projection.weight"
        ]
        .detach()
        .cpu()
    )

    if weights.shape != (512, 3 * 32 * 32):
        raise ValueError(
            f"Unexpected weight shape in {path}: "
            f"{tuple(weights.shape)}"
        )

    return weights.reshape(512, 3, 32, 32)


def normalized_energy(weight):
    energy = weight.square().sum(dim=0).numpy()

    maximum = float(energy.max())

    if maximum > 0:
        energy /= maximum

    return energy


def create_atlas(
    data,
    assignments,
    weights_by_seed,
    start_rank,
    end_rank,
    output_name,
):
    template_order = data[
        f"{CONDITION}_template_order"
    ].astype(int)

    template_energy = data[
        f"{CONDITION}_template_energy"
    ]

    condition_rows = assignments[
        assignments["condition"] == CONDITION
    ].copy()

    family_count = end_rank - start_rank + 1

    fig, axes = plt.subplots(
        family_count,
        1 + len(SEEDS),
        figsize=(12.8, 14.4),
    )

    fig.suptitle(
        "Learned local template families and their strongest matches",
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )

    column_titles = [
        "Learned template",
        "Seed 43",
        "Seed 44",
        "Seed 45",
        "Seed 46",
        "Seed 47",
    ]

    for column, title in enumerate(column_titles):
        axes[0, column].set_title(
            title,
            fontsize=10.5,
            fontweight="semibold",
            pad=8,
        )

    for display_row, family_rank in enumerate(
        range(start_rank, end_rank + 1)
    ):
        template_id = int(
            template_order[family_rank - 1]
        )

        family_rows = condition_rows[
            condition_rows["template_id"]
            == template_id
        ].copy()

        usage = (
            len(family_rows)
            / (len(SEEDS) * 512)
        )

        template = (
            template_energy[template_id]
            .astype(float)
        )

        template /= max(
            float(template.max()),
            1e-12,
        )

        ax_template = axes[
            display_row,
            0,
        ]

        ax_template.imshow(
            template,
            cmap="viridis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax_template.set_xticks([])
        ax_template.set_yticks([])

        for spine in ax_template.spines.values():
            spine.set_linewidth(1.3)

        ax_template.set_ylabel(
            f"Family {family_rank}\n"
            f"{100 * usage:.1f}% of neurons",
            fontsize=9.2,
            fontweight="semibold",
            rotation=0,
            labelpad=48,
            va="center",
        )

        for seed_column, seed in enumerate(
            SEEDS,
            start=1,
        ):
            ax = axes[
                display_row,
                seed_column,
            ]

            seed_rows = family_rows[
                family_rows["seed"] == seed
            ].copy()

            if seed_rows.empty:
                ax.axis("off")
                continue

            best_index = (
                seed_rows[
                    "squared_template_match"
                ]
                .idxmax()
            )

            best = seed_rows.loc[
                best_index
            ]

            neuron = int(
                best["neuron_index"]
            )

            patch_row = int(
                best["patch_row"]
            )

            patch_column = int(
                best["patch_column"]
            )

            weight = weights_by_seed[
                seed
            ][neuron]

            image = normalized_energy(
                weight
            )

            ax.imshow(
                image,
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

            ax.add_patch(
                Rectangle(
                    (
                        patch_column - 0.5,
                        patch_row - 0.5,
                    ),
                    PATCH_SIZE,
                    PATCH_SIZE,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.35,
                )
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.94,
        bottom=0.025,
        wspace=0.055,
        hspace=0.08,
    )

    output = (
        FIGURE_DIR
        / output_name
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            ASSIGNMENTS_PATH
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.load(DATA_PATH)

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH
    )

    weights_by_seed = {
        seed: load_weights(seed)
        for seed in SEEDS
    }

    create_atlas(
        data=data,
        assignments=assignments,
        weights_by_seed=weights_by_seed,
        start_rank=1,
        end_rank=8,
        output_name=(
            "figure_template_family_atlas_01_08"
        ),
    )

    create_atlas(
        data=data,
        assignments=assignments,
        weights_by_seed=weights_by_seed,
        start_rank=9,
        end_rank=16,
        output_name=(
            "figure_template_family_atlas_09_16"
        ),
    )

    print("Two complete template-family atlases created.")


if __name__ == "__main__":
    main()
