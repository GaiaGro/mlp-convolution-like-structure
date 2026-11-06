from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.structural import (
    extract_maximum_energy_patches,
    fit_sign_invariant_spherical_kmeans,
)


CHECKPOINT_DIR = Path("results/final/training/checkpoints")
CONFIG_PATH = Path("configs/structural_analysis.yaml")

OUTPUT_DIR = Path(
    "results/structural/spatial_template_reuse"
)

CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]


SEEDS = [43, 44, 45, 46, 47]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    return yaml.safe_load(
        path.read_text(encoding="utf-8")
    )


def find_image_shape(config: dict) -> tuple[int, int, int]:
    def search(value):
        if isinstance(value, dict):
            if "image_shape" in value:
                return value["image_shape"]

            for child in value.values():
                result = search(child)
                if result is not None:
                    return result

        return None

    result = search(config)

    if result is None:
        raise KeyError(
            "Could not find image_shape in structural config."
        )

    shape = tuple(int(x) for x in result)

    if len(shape) != 3:
        raise ValueError(
            f"Expected image_shape with 3 values, got {shape}."
        )

    return shape


def load_first_layer_weights(
    condition: str,
    seed: int,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    run_name = f"mlp_{condition}_seed{seed}"

    checkpoint_path = (
        CHECKPOINT_DIR / f"{run_name}_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint["model_state_dict"]
    parameter_name = "input_projection.weight"

    if parameter_name not in state:
        raise KeyError(
            f"{parameter_name} missing from {checkpoint_path}"
        )

    weights = (
        state[parameter_name]
        .detach()
        .clone()
        .cpu()
    )

    expected_inputs = int(np.prod(image_shape))

    if (
        weights.ndim != 2
        or weights.shape[1] != expected_inputs
    ):
        raise ValueError(
            f"Unexpected first-layer shape in {checkpoint_path}: "
            f"{tuple(weights.shape)}"
        )

    return weights.reshape(
        weights.shape[0],
        *image_shape,
    )


def match_templates(
    weights: torch.Tensor,
    templates: torch.Tensor,
    patch_size: int,
):
    """
    Mirror the matching logic of translated_template_scores,
    while retaining the winning template and spatial location.
    """
    neurons, channels, height, width = weights.shape

    expected_dimension = (
        channels * patch_size * patch_size
    )

    patches = (
        weights.unfold(2, patch_size, 1)
        .unfold(3, patch_size, 1)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(
            neurons,
            -1,
            expected_dimension,
        )
    )

    similarities = torch.einsum(
        "npd,kd->npk",
        patches,
        templates,
    )

    squared_similarities = similarities.square()

    flattened = squared_similarities.flatten(
        start_dim=1
    )

    best_scores, best_indices = flattened.max(
        dim=1
    )

    template_count = templates.shape[0]

    position_indices = torch.div(
        best_indices,
        template_count,
        rounding_mode="floor",
    )

    template_indices = best_indices.remainder(
        template_count
    )

    position_width = width - patch_size + 1

    rows = torch.div(
        position_indices,
        position_width,
        rounding_mode="floor",
    )

    columns = position_indices.remainder(
        position_width
    )

    center_offset = patch_size // 2

    center_rows = rows + center_offset
    center_columns = columns + center_offset

    return {
        "template_indices": template_indices,
        "rows": rows,
        "columns": columns,
        "center_rows": center_rows,
        "center_columns": center_columns,
        "best_scores": best_scores,
    }


def analyze_condition(
    condition: str,
    image_shape: tuple[int, int, int],
    patch_size: int,
    template_count: int,
    initializations: int,
    maximum_iterations: int,
    tolerance: float,
    template_seed: int,
):
    print(f"Analyzing {condition}...")

    weights_by_seed = {}
    fitting_samples = []

    for seed in SEEDS:
        weights = load_first_layer_weights(
            condition=condition,
            seed=seed,
            image_shape=image_shape,
        )

        weights_by_seed[seed] = weights

        maximum_patches, _, _ = (
            extract_maximum_energy_patches(
                weights,
                patch_size,
            )
        )

        fitting_samples.append(
            maximum_patches.flatten(start_dim=1)
        )

    fitting_samples = torch.cat(
        fitting_samples,
        dim=0,
    )

    expected_sample_count = (
        len(SEEDS)
        * next(iter(weights_by_seed.values())).shape[0]
    )

    if fitting_samples.shape[0] != expected_sample_count:
        raise RuntimeError(
            "Unexpected number of fitting samples."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(template_seed)

    templates = fit_sign_invariant_spherical_kmeans(
        samples=fitting_samples,
        template_count=template_count,
        initializations=initializations,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        generator=generator,
    )

    norms = templates.norm(dim=1)

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    ):
        raise RuntimeError(
            "Fitted templates are not unit norm."
        )

    height = image_shape[1]
    width = image_shape[2]

    seed_maps = torch.zeros(
        len(SEEDS),
        template_count,
        height,
        width,
        dtype=torch.float32,
    )

    usage = torch.zeros(
        len(SEEDS),
        template_count,
        dtype=torch.float32,
    )

    assignment_rows = []

    for seed_index, seed in enumerate(SEEDS):
        weights = weights_by_seed[seed]

        matches = match_templates(
            weights=weights,
            templates=templates,
            patch_size=patch_size,
        )

        template_indices = matches[
            "template_indices"
        ]

        center_rows = matches["center_rows"]
        center_columns = matches["center_columns"]

        neuron_count = weights.shape[0]

        for neuron_index in range(neuron_count):
            template_id = int(
                template_indices[neuron_index].item()
            )

            center_row = int(
                center_rows[neuron_index].item()
            )

            center_column = int(
                center_columns[neuron_index].item()
            )

            seed_maps[
                seed_index,
                template_id,
                center_row,
                center_column,
            ] += 1.0 / neuron_count

            usage[
                seed_index,
                template_id,
            ] += 1.0 / neuron_count

            assignment_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "neuron_index": neuron_index,
                    "template_id": template_id,
                    "patch_row": int(
                        matches["rows"][
                            neuron_index
                        ].item()
                    ),
                    "patch_column": int(
                        matches["columns"][
                            neuron_index
                        ].item()
                    ),
                    "patch_center_row": center_row,
                    "patch_center_column": center_column,
                    "squared_template_match": float(
                        matches["best_scores"][
                            neuron_index
                        ].item()
                    ),
                }
            )

        if not torch.allclose(
            usage[seed_index].sum(),
            torch.tensor(1.0),
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Template usage does not sum to one "
                f"for {condition}, seed {seed}."
            )

        if not torch.allclose(
            seed_maps[seed_index].sum(),
            torch.tensor(1.0),
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Spatial map does not sum to one "
                f"for {condition}, seed {seed}."
            )

    mean_usage = usage.mean(dim=0)

    template_order = torch.argsort(
        mean_usage,
        descending=True,
    )

    rank_lookup = {
        int(template_id.item()): rank + 1
        for rank, template_id
        in enumerate(template_order)
    }

    for row in assignment_rows:
        row["template_usage_rank"] = (
            rank_lookup[row["template_id"]]
        )

    summary_rows = []

    for template_id in range(template_count):
        values = usage[:, template_id]

        summary_rows.append(
            {
                "condition": condition,
                "template_id": template_id,
                "template_usage_rank": (
                    rank_lookup[template_id]
                ),
                "mean_usage_fraction": float(
                    values.mean().item()
                ),
                "std_usage_fraction": float(
                    values.std(
                        unbiased=False
                    ).item()
                ),
            }
        )

    template_energy = (
        templates.reshape(
            template_count,
            image_shape[0],
            patch_size,
            patch_size,
        )
        .square()
        .sum(dim=1)
    )

    mean_spatial_maps = seed_maps.mean(dim=0)

    return {
        "templates": templates,
        "template_energy": template_energy,
        "seed_maps": seed_maps,
        "mean_spatial_maps": mean_spatial_maps,
        "usage": usage,
        "template_order": template_order,
        "assignments": assignment_rows,
        "summary": summary_rows,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    structural_config = load_yaml(CONFIG_PATH)

    image_shape = find_image_shape(
        structural_config
    )

    template_config = structural_config[
        "template_reuse"
    ]

    patch_size = int(
        template_config["patch_size"]
    )

    template_count = int(
        template_config["templates"]
    )

    initializations = int(
        template_config["initializations"]
    )

    maximum_iterations = int(
        template_config["maximum_iterations"]
    )

    tolerance = float(
        template_config["tolerance"]
    )

    template_seed = int(
        template_config["seed"]
    )

    if template_count != 16:
        raise ValueError(
            "This visualization expects the frozen "
            f"16-template configuration, found {template_count}."
        )

    if patch_size != 7:
        raise ValueError(
            "This visualization expects the frozen "
            f"7x7 primary patch size, found {patch_size}."
        )

    results = {}

    all_assignments = []
    all_summaries = []

    for condition in CONDITIONS:
        result = analyze_condition(
            condition=condition,
            image_shape=image_shape,
            patch_size=patch_size,
            template_count=template_count,
            initializations=initializations,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            template_seed=template_seed,
        )

        results[condition] = result

        all_assignments.extend(
            result["assignments"]
        )

        all_summaries.extend(
            result["summary"]
        )

    assignments = pd.DataFrame(
        all_assignments
    )

    summaries = pd.DataFrame(
        all_summaries
    )

    expected_assignments = (
        len(CONDITIONS)
        * len(SEEDS)
        * 512
    )

    if len(assignments) != expected_assignments:
        raise RuntimeError(
            "Unexpected number of neuron assignments: "
            f"{len(assignments)} != {expected_assignments}"
        )

    assignments.to_csv(
        OUTPUT_DIR
        / "spatial_template_reuse_assignments.csv",
        index=False,
    )

    summaries.to_csv(
        OUTPUT_DIR
        / "spatial_template_reuse_summary.csv",
        index=False,
    )

    arrays = {}

    for condition, result in results.items():
        arrays[
            f"{condition}_templates"
        ] = result["templates"].numpy()

        arrays[
            f"{condition}_template_energy"
        ] = result["template_energy"].numpy()

        arrays[
            f"{condition}_seed_spatial_maps"
        ] = result["seed_maps"].numpy()

        arrays[
            f"{condition}_mean_spatial_maps"
        ] = result[
            "mean_spatial_maps"
        ].numpy()

        arrays[
            f"{condition}_usage"
        ] = result["usage"].numpy()

        arrays[
            f"{condition}_template_order"
        ] = result[
            "template_order"
        ].numpy()

    np.savez_compressed(
        OUTPUT_DIR
        / "spatial_template_reuse_data.npz",
        **arrays,
    )

    print()
    print("Spatial template reuse analysis complete.")
    print(
        f"Assignments: {len(assignments)} "
        f"(expected {expected_assignments})"
    )
    print(
        f"Results: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
