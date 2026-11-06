import argparse
from pathlib import Path

import pandas as pd
import torch

from scripts.structural_analysis import (
    build_spatial_permutation,
    load_checkpoint_weights,
    load_yaml,
    random_initialization_weights,
    validate_run,
)
from src.evaluation.structural import (
    cross_validated_template_reuse,
    shuffle_spatial_weights,
    weights_in_original_image_coordinates,
)


VARIANTS = {
    "patch5_templates16": {
        "patch_size": 5,
        "template_count": 16,
    },
    "patch9_templates16": {
        "patch_size": 9,
        "template_count": 16,
    },
    "patch7_templates8": {
        "patch_size": 7,
        "template_count": 8,
    },
    "patch7_templates32": {
        "patch_size": 7,
        "template_count": 32,
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Structural template-reuse sensitivity analysis."
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[43, 44, 45, 46, 47],
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        required=True,
    )
    parser.add_argument(
        "--training-directory",
        type=Path,
        default=Path("results/final/training"),
    )
    parser.add_argument(
        "--pixel-directory",
        type=Path,
        default=Path("results/structural/pixel_permutation"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/structural/sensitivity"),
    )
    return parser.parse_args()


def validate_primary_config(config):
    template = config["template_reuse"]

    if int(template["patch_size"]) != 7:
        raise ValueError(
            "Primary patch size has changed; expected 7."
        )

    if int(template["templates"]) != 16:
        raise ValueError(
            "Primary template count has changed; expected 16."
        )

    if int(config["locality"]["primary_window_size"]) != 7:
        raise ValueError(
            "Primary locality window has changed; expected 7."
        )


def template_arguments(
    analysis_config,
    variant,
    seed,
):
    template = analysis_config["template_reuse"]

    return {
        "patch_size": int(variant["patch_size"]),
        "template_count": int(variant["template_count"]),
        "folds": int(template["folds"]),
        "repetitions": int(template["repetitions"]),
        "initializations": int(template["initializations"]),
        "maximum_iterations": int(
            template["maximum_iterations"]
        ),
        "tolerance": float(template["tolerance"]),
        "seed": int(seed),
    }


def analyze_weight_set(
    weights,
    metadata,
    analysis_config,
    variant_name,
    variant,
    template_seed,
):
    reuse = cross_validated_template_reuse(
        weights=weights,
        **template_arguments(
            analysis_config,
            variant,
            template_seed,
        ),
    )

    row = dict(metadata)
    row["sensitivity_id"] = variant_name
    row["patch_size"] = int(variant["patch_size"])
    row["templates"] = int(variant["template_count"])

    output_names = {
        "conditional_shape_score":
            "template_conditional_score",
        "total_energy_reconstruction_fraction":
            "template_total_reconstruction",
        "selected_patch_energy_fraction":
            "template_selected_patch_energy",
        "maximum_patch_energy_fraction":
            "maximum_patch_energy_fraction",
    }

    for source_name, output_name in output_names.items():
        values = reuse[source_name]

        row[f"mean_{output_name}"] = float(
            values.mean().item()
        )
        row[f"std_{output_name}"] = float(
            values.std(unbiased=False).item()
        )

    return row


def partial_is_complete(path, variant_name):
    if not path.exists():
        return False

    try:
        table = pd.read_csv(path)
    except Exception:
        return False

    return (
        len(table) == 1
        and table.iloc[0]["sensitivity_id"]
        == variant_name
    )


def analyze_and_save(
    weights,
    metadata,
    analysis_config,
    variant_name,
    variant,
    template_seed,
    partial_directory,
):
    analysis_id = metadata["analysis_id"]

    path = partial_directory / f"{analysis_id}_run.csv"

    if partial_is_complete(path, variant_name):
        print(
            f"Skipping completed analysis: {analysis_id}",
            flush=True,
        )
        return

    print(
        f"Analyzing: {variant_name} | {analysis_id}",
        flush=True,
    )

    row = analyze_weight_set(
        weights,
        metadata,
        analysis_config,
        variant_name,
        variant,
        template_seed,
    )

    temporary = path.with_suffix(".csv.tmp")

    pd.DataFrame([row]).to_csv(
        temporary,
        index=False,
    )
    temporary.replace(path)

    print(
        f"Completed: {variant_name} | {analysis_id}",
        flush=True,
    )


def combine_results(
    partial_directory,
    output_directory,
    seed,
    variant_name,
    expected_analyses,
):
    paths = sorted(
        partial_directory.glob("*_run.csv")
    )

    if len(paths) != expected_analyses:
        raise ValueError(
            f"Expected {expected_analyses} partials for "
            f"{variant_name}, seed {seed}; "
            f"found {len(paths)}."
        )

    table = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

    if len(table) != expected_analyses:
        raise ValueError(
            "Combined sensitivity table has invalid size."
        )

    if table["analysis_id"].duplicated().any():
        raise ValueError(
            "Duplicate analysis_id in sensitivity results."
        )

    if set(table["sensitivity_id"]) != {variant_name}:
        raise ValueError(
            "Unexpected sensitivity variant in results."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{variant_name}_seed{seed}.csv"
    )

    table.sort_values(
        [
            "source",
            "coordinate_system",
            "condition",
            "shuffle_index",
        ]
    ).to_csv(
        output_path,
        index=False,
    )

    return table


def main():
    args = parse_arguments()

    analysis_config = load_yaml(
        Path("configs/structural_analysis.yaml")
    )
    control_config = load_yaml(
        Path("configs/structural_controls.yaml")
    )
    experiment_config = load_yaml(
        Path("configs/experiment.yaml")
    )

    validate_primary_config(analysis_config)

    variant = VARIANTS[args.variant]

    pixel_config = control_config["pixel_permutation"]
    conditions = list(pixel_config["conditions"])

    expected_conditions = [
        "baseline",
        "exposure",
        "consistency",
        "augmentation",
        "augmentation_consistency",
    ]

    if conditions != expected_conditions:
        raise ValueError(
            "Unexpected structural-control conditions."
        )

    if args.seed not in {
        int(x) for x in pixel_config["final_seeds"]
    }:
        raise ValueError(
            "Seed is not a predefined final seed."
        )

    image_shape = tuple(
        int(x)
        for x in analysis_config["first_layer"][
            "image_shape"
        ]
    )

    if image_shape != (3, 32, 32):
        raise ValueError(
            f"Unexpected image shape: {image_shape}."
        )

    permutation_seed = int(
        pixel_config["spatial_seed"]
    )

    shuffle_config = analysis_config["controls"][
        "spatial_shuffle"
    ]

    shuffle_repetitions = int(
        shuffle_config["repetitions"]
    )

    if shuffle_repetitions != 20:
        raise ValueError(
            "Expected exactly 20 spatial shuffles."
        )

    shuffle_seed = int(
        shuffle_config["seed"]
    )

    template_base_seed = int(
        analysis_config["template_reuse"]["seed"]
    )

    expected_neurons = int(
        experiment_config["mlp"]["hidden_width"]
    )

    if expected_neurons != 512:
        raise ValueError(
            f"Unexpected MLP width: {expected_neurons}."
        )

    expected_analyses = 1 + len(conditions) * (
        1 + shuffle_repetitions + 2
    )

    if expected_analyses != 116:
        raise ValueError(
            f"Expected 116 analyses, got "
            f"{expected_analyses}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")

    partial_directory = (
        args.output_directory
        / "partials"
        / args.variant
        / f"seed{args.seed}"
    )
    partial_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_directory = (
        args.output_directory / "combined"
    )

    print(f"Device: {device}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Variant: {args.variant}", flush=True)
    print(
        f"Patch size: {variant['patch_size']}",
        flush=True,
    )
    print(
        f"Templates: {variant['template_count']}",
        flush=True,
    )
    print(
        f"Expected analyses: {expected_analyses}",
        flush=True,
    )

    for condition_index, condition in enumerate(
        conditions
    ):
        primary_run_name = (
            f"mlp_{condition}_seed{args.seed}"
        )

        primary_config_path = (
            args.training_directory
            / "configs"
            / f"{primary_run_name}.yaml"
        )

        primary_run, primary_weights = (
            load_checkpoint_weights(
                config_path=primary_config_path,
                checkpoint_directory=(
                    args.training_directory
                    / "checkpoints"
                ),
                image_shape=image_shape,
            )
        )

        validate_run(
            primary_run,
            condition,
            args.seed,
            None,
        )

        primary_weights = primary_weights.to(device)

        primary_metadata = {
            "analysis_id":
                f"trained_{condition}_seed{args.seed}",
            "source": "trained",
            "coordinate_system": "original_image",
            "condition": condition,
            "seed": args.seed,
            "run_name": primary_run_name,
            "parent_run_name": "",
            "shuffle_index": -1,
        }

        analyze_and_save(
            primary_weights,
            primary_metadata,
            analysis_config,
            args.variant,
            variant,
            template_base_seed
            + args.seed * 1_000_000
            + condition_index * 10_000
            + 1,
            partial_directory,
        )

        for shuffle_index in range(
            shuffle_repetitions
        ):
            generator = torch.Generator(
                device="cpu"
            )
            generator.manual_seed(
                shuffle_seed
                + args.seed * 10_000
                + condition_index * 100
                + shuffle_index
            )

            shuffled_weights = (
                shuffle_spatial_weights(
                    primary_weights,
                    generator,
                )
            )

            shuffle_metadata = {
                "analysis_id": (
                    f"shuffle_{condition}_seed"
                    f"{args.seed}_rep{shuffle_index:02d}"
                ),
                "source": "spatial_shuffle",
                "coordinate_system": "original_image",
                "condition": condition,
                "seed": args.seed,
                "run_name": (
                    f"shuffle_{primary_run_name}_"
                    f"rep{shuffle_index:02d}"
                ),
                "parent_run_name": primary_run_name,
                "shuffle_index": shuffle_index,
            }

            analyze_and_save(
                shuffled_weights,
                shuffle_metadata,
                analysis_config,
                args.variant,
                variant,
                template_base_seed
                + args.seed * 1_000_000
                + condition_index * 10_000
                + 100
                + shuffle_index,
                partial_directory,
            )

        pixel_run_name = (
            f"mlp_pixelperm_{condition}_seed"
            f"{args.seed}"
        )

        pixel_config_path = (
            args.pixel_directory
            / "configs"
            / f"{pixel_run_name}.yaml"
        )

        pixel_run, pixel_weights = (
            load_checkpoint_weights(
                config_path=pixel_config_path,
                checkpoint_directory=(
                    args.pixel_directory
                    / "checkpoints"
                ),
                image_shape=image_shape,
            )
        )

        validate_run(
            pixel_run,
            condition,
            args.seed,
            permutation_seed,
        )

        pixel_weights = pixel_weights.to(device)

        model_metadata = {
            "analysis_id": (
                f"pixel_model_{condition}_seed"
                f"{args.seed}"
            ),
            "source": "pixel_permutation",
            "coordinate_system": "model",
            "condition": condition,
            "seed": args.seed,
            "run_name": pixel_run_name,
            "parent_run_name": "",
            "shuffle_index": -1,
        }

        analyze_and_save(
            pixel_weights,
            model_metadata,
            analysis_config,
            args.variant,
            variant,
            template_base_seed
            + args.seed * 1_000_000
            + condition_index * 10_000
            + 500,
            partial_directory,
        )

        pixel_count = (
            image_shape[1] * image_shape[2]
        )

        permutation = build_spatial_permutation(
            permutation_seed,
            pixel_count,
        )

        image_weights = (
            weights_in_original_image_coordinates(
                pixel_weights,
                permutation,
            )
        )

        image_metadata = {
            "analysis_id": (
                f"pixel_image_{condition}_seed"
                f"{args.seed}"
            ),
            "source": "pixel_permutation",
            "coordinate_system": "original_image",
            "condition": condition,
            "seed": args.seed,
            "run_name": pixel_run_name,
            "parent_run_name": "",
            "shuffle_index": -1,
        }

        analyze_and_save(
            image_weights,
            image_metadata,
            analysis_config,
            args.variant,
            variant,
            template_base_seed
            + args.seed * 1_000_000
            + condition_index * 10_000
            + 600,
            partial_directory,
        )

    random_weights = random_initialization_weights(
        experiment_config,
        args.seed,
        image_shape,
    ).to(device)

    random_metadata = {
        "analysis_id": (
            f"random_initialization_seed{args.seed}"
        ),
        "source": "random_initialization",
        "coordinate_system": "original_image",
        "condition": "not_applicable",
        "seed": args.seed,
        "run_name": (
            f"mlp_random_initialization_seed"
            f"{args.seed}"
        ),
        "parent_run_name": "",
        "shuffle_index": -1,
    }

    analyze_and_save(
        random_weights,
        random_metadata,
        analysis_config,
        args.variant,
        variant,
        template_base_seed
        + args.seed * 1_000_000
        + 900_000,
        partial_directory,
    )

    table = combine_results(
        partial_directory,
        combined_directory,
        args.seed,
        args.variant,
        expected_analyses,
    )

    if len(table) != 116:
        raise AssertionError(
            "Sensitivity output must contain 116 rows."
        )

    print(
        f"Completed sensitivity: "
        f"{args.variant}, seed {args.seed}",
        flush=True,
    )
    print(
        f"Results saved to: {combined_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
