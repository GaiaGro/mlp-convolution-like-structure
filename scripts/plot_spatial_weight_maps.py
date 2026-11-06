from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "semibold"
import numpy as np
import torch
import yaml

from scripts.structural_analysis import (
    build_spatial_permutation,
    load_checkpoint_weights,
    random_initialization_weights,
)
from src.evaluation.structural import (
    maximum_window_energy_fraction,
    weights_in_original_image_coordinates,
)


ROOT = Path(".")
EXPERIMENT_CONFIG = ROOT / "configs/experiment.yaml"

TRAINING_DIR = ROOT / "results/final/training"
PIXEL_DIR = ROOT / "results/structural/pixel_permutation"

OUTPUT_DIR = ROOT / "figures/spatial_weight_maps"

SEEDS = [43, 44, 45, 46, 47]
WINDOW_SIZE = 7

PROGRESSION_CONDITIONS = [
    ("random", "Random initialization"),
    ("baseline", "Baseline"),
    ("augmentation", "Augmentation"),
    (
        "augmentation_consistency",
        "Augmentation + consistency",
    ),
]

GEOMETRY_CONDITIONS = [
    ("baseline", "Baseline"),
    (
        "augmentation_consistency",
        "Augmentation + consistency",
    ),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_experiment_config() -> dict:
    require(
        EXPERIMENT_CONFIG.exists(),
        f"Missing {EXPERIMENT_CONFIG}",
    )
    return yaml.safe_load(
        EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    )


def load_trained_weights(
    condition: str,
    seed: int,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    run_name = f"mlp_{condition}_seed{seed}"

    config_path = (
        TRAINING_DIR
        / "configs"
        / f"{run_name}.yaml"
    )

    _, weights = load_checkpoint_weights(
        config_path=config_path,
        checkpoint_directory=(
            TRAINING_DIR / "checkpoints"
        ),
        image_shape=image_shape,
    )

    return weights.cpu()


def load_pixel_weights(
    condition: str,
    seed: int,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    run_name = (
        f"mlp_pixelperm_{condition}_seed{seed}"
    )

    config_path = (
        PIXEL_DIR
        / "configs"
        / f"{run_name}.yaml"
    )

    _, weights = load_checkpoint_weights(
        config_path=config_path,
        checkpoint_directory=(
            PIXEL_DIR / "checkpoints"
        ),
        image_shape=image_shape,
    )

    return weights.cpu()


def maximum_patch_density(
    weights: torch.Tensor,
    window_size: int,
) -> np.ndarray:
    """
    For each first-layer neuron, find its maximum-energy
    window and count which image pixels are covered by it.

    Output is normalized so that 1.0 would mean every
    neuron places its maximum-energy patch over that pixel.
    """
    _, locations = maximum_window_energy_fraction(
        weights,
        window_size,
    )

    height = weights.shape[-2]
    width = weights.shape[-1]

    density = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    for row, col in locations.cpu().numpy():
        row = int(row)
        col = int(col)

        density[
            row : row + window_size,
            col : col + window_size,
        ] += 1.0

    density /= float(weights.shape[0])

    return density


def aggregate_density(
    weight_sets: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    maps = np.stack(
        [
            maximum_patch_density(
                weights,
                WINDOW_SIZE,
            )
            for weights in weight_sets
        ],
        axis=0,
    )

    return maps.mean(axis=0), maps.std(axis=0)


def progression_maps(
    experiment: dict,
    image_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    result = {}

    random_weights = [
        random_initialization_weights(
            experiment_config=experiment,
            seed=seed,
            image_shape=image_shape,
        ).cpu()
        for seed in SEEDS
    ]

    result["random"], _ = aggregate_density(
        random_weights
    )

    for condition, _ in PROGRESSION_CONDITIONS:
        if condition == "random":
            continue

        weights = [
            load_trained_weights(
                condition=condition,
                seed=seed,
                image_shape=image_shape,
            )
            for seed in SEEDS
        ]

        result[condition], _ = aggregate_density(
            weights
        )

    return result


def geometry_maps(
    experiment: dict,
    image_shape: tuple[int, int, int],
) -> dict[tuple[str, str], np.ndarray]:
    result = {}

    pixel_config = yaml.safe_load(
        (
            ROOT
            / "configs/structural_controls.yaml"
        ).read_text(encoding="utf-8")
    )

    permutation_seed = int(
        pixel_config["pixel_permutation"][
            "spatial_seed"
        ]
    )

    pixel_count = (
        image_shape[1] * image_shape[2]
    )

    permutation = build_spatial_permutation(
        permutation_seed,
        pixel_count,
    )

    for condition, _ in GEOMETRY_CONDITIONS:
        trained_sets = []
        pixel_model_sets = []
        pixel_image_sets = []

        for seed in SEEDS:
            trained = load_trained_weights(
                condition=condition,
                seed=seed,
                image_shape=image_shape,
            )

            pixel_model = load_pixel_weights(
                condition=condition,
                seed=seed,
                image_shape=image_shape,
            )

            pixel_image = (
                weights_in_original_image_coordinates(
                    pixel_model,
                    permutation,
                )
            )

            trained_sets.append(trained)
            pixel_model_sets.append(pixel_model)
            pixel_image_sets.append(
                pixel_image.cpu()
            )

        result[
            (condition, "trained")
        ], _ = aggregate_density(trained_sets)

        result[
            (condition, "pixel_model")
        ], _ = aggregate_density(
            pixel_model_sets
        )

        result[
            (condition, "pixel_image")
        ], _ = aggregate_density(
            pixel_image_sets
        )

    return result


def save_progression_figure(
    maps: dict[str, np.ndarray],
) -> None:
    values = np.stack(
        [
            maps[key]
            for key, _ in PROGRESSION_CONDITIONS
        ],
        axis=0,
    )

    vmin = float(values.min())
    vmax = float(values.max())

    fig, axes = plt.subplots(
        1,
        len(PROGRESSION_CONDITIONS),
        figsize=(13.5, 3.8),
        constrained_layout=True,
    )

    image = None

    for ax, (key, title) in zip(
        axes,
        PROGRESSION_CONDITIONS,
    ):
        image = ax.imshow(
            maps[key],
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="upper",
        )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Image x-coordinate")
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])

    axes[0].set_ylabel("Image y-coordinate")

    for ax in axes[1:]:
        ax.set_yticklabels([])

    cbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        pad=0.02,
    )

    cbar.set_label(
        "Fraction of neurons whose "
        "maximum-energy 7×7 patch covers pixel"
    )

    fig.suptitle(
        "Emergence of spatial concentration "
        "in first-layer MLP weights",
        fontsize=14,
    )

    png = (
        OUTPUT_DIR
        / "figure_spatial_concentration_conditions.png"
    )
    pdf = (
        OUTPUT_DIR
        / "figure_spatial_concentration_conditions.pdf"
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


def save_geometry_figure(
    maps: dict[tuple[str, str], np.ndarray],
) -> None:
    column_keys = [
        ("trained", "Trained"),
        (
            "pixel_model",
            "Pixel permutation\nmodel coordinates",
        ),
        (
            "pixel_image",
            "Pixel permutation\noriginal-image coordinates",
        ),
    ]

    all_maps = np.stack(
        list(maps.values()),
        axis=0,
    )

    vmin = float(all_maps.min())
    vmax = float(all_maps.max())

    fig, axes = plt.subplots(
        len(GEOMETRY_CONDITIONS),
        len(column_keys),
        figsize=(10.5, 7),
        constrained_layout=True,
    )

    image = None

    for row_idx, (condition, row_label) in enumerate(
        GEOMETRY_CONDITIONS
    ):
        for col_idx, (map_key, title) in enumerate(
            column_keys
        ):
            ax = axes[row_idx, col_idx]

            image = ax.imshow(
                maps[(condition, map_key)],
                cmap="Blues",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="upper",
            )

            if row_idx == 0:
                ax.set_title(
                    title,
                    fontsize=11,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    f"{row_label}\nImage y-coordinate"
                )
            else:
                ax.set_yticklabels([])

            if row_idx == len(
                GEOMETRY_CONDITIONS
            ) - 1:
                ax.set_xlabel(
                    "Image x-coordinate"
                )

            ax.set_xticks([0, 8, 16, 24, 31])
            ax.set_yticks([0, 8, 16, 24, 31])

    cbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        pad=0.02,
    )

    cbar.set_label(
        "Fraction of neurons whose "
        "maximum-energy 7×7 patch covers pixel"
    )

    fig.suptitle(
        "Spatial geometry under pixel permutation",
        fontsize=14,
    )

    png = (
        OUTPUT_DIR
        / "figure_pixel_permutation_geometry.png"
    )
    pdf = (
        OUTPUT_DIR
        / "figure_pixel_permutation_geometry.pdf"
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


def save_arrays(
    progression: dict[str, np.ndarray],
    geometry: dict[
        tuple[str, str],
        np.ndarray,
    ],
) -> None:
    arrays = {}

    for key, value in progression.items():
        arrays[
            f"progression_{key}"
        ] = value

    for (
        condition,
        coordinate,
    ), value in geometry.items():
        arrays[
            f"geometry_{condition}_{coordinate}"
        ] = value

    data_dir = ROOT / "results/structural/spatial_weight_maps"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        data_dir / "spatial_weight_map_data.npz",
        **arrays,
    )


def main() -> None:
    experiment = load_experiment_config()

    image_shape = tuple(
        int(v)
        for v in experiment["model"]["image_shape"]
    ) if (
        "model" in experiment
        and "image_shape" in experiment["model"]
    ) else (3, 32, 32)

    require(
        image_shape == (3, 32, 32),
        f"Unexpected image shape: {image_shape}",
    )

    require(
        (ROOT / "configs/structural_controls.yaml").exists(),
        "Missing configs/structural_controls.yaml",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Building progression maps...")

    progression = progression_maps(
        experiment=experiment,
        image_shape=image_shape,
    )

    print("Building pixel-permutation geometry maps...")

    geometry = geometry_maps(
        experiment=experiment,
        image_shape=image_shape,
    )

    save_progression_figure(progression)
    save_geometry_figure(geometry)
    save_arrays(progression, geometry)

    print()
    print("DONE")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print(
        "- figure_spatial_concentration_conditions.png"
    )
    print(
        "- figure_pixel_permutation_geometry.png"
    )
    print(
        "- matching PDF versions"
    )
    print(
        "- spatial_weight_map_data.npz"
    )


if __name__ == "__main__":
    main()
