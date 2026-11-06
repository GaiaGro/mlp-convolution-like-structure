import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.evaluation.structural import (
    cross_validated_template_reuse,
    maximum_window_energy_fraction,
    shuffle_spatial_weights,
    weights_in_original_image_coordinates,
)
from src.models import build_model
from src.utils import set_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze first-layer MLP structure."
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[43, 44, 45, 46, 47],
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
        default=Path(
            "results/structural/pixel_permutation"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/structural/analysis"),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    return yaml.safe_load(
        path.read_text(encoding="utf-8")
    )


def load_checkpoint_weights(
    config_path: Path,
    checkpoint_directory: Path,
    image_shape: tuple[int, int, int],
) -> tuple[dict, torch.Tensor]:
    run = load_yaml(config_path)

    if run["model"] != "mlp":
        raise ValueError(
            f"Expected an MLP run: {config_path}"
        )

    run_name = run["run_name"]
    checkpoint_path = (
        checkpoint_directory / f"{run_name}_best.pt"
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

    weights = state[parameter_name].detach().clone()
    expected_inputs = (
        image_shape[0] * image_shape[1] * image_shape[2]
    )

    if weights.ndim != 2 or weights.shape[1] != expected_inputs:
        raise ValueError(
            f"Unexpected first-layer shape in {checkpoint_path}"
        )

    weights = weights.reshape(
        weights.shape[0],
        *image_shape,
    )

    return run, weights


def random_initialization_weights(
    experiment_config: dict,
    seed: int,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    set_seed(seed)

    model = build_model(
        "mlp",
        experiment_config,
        pixel_permutation_seed=None,
    )

    weights = (
        model.input_projection.weight
        .detach()
        .clone()
        .reshape(
            model.hidden_width,
            *image_shape,
        )
    )

    return weights


def build_spatial_permutation(
    seed: int,
    pixel_count: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return torch.randperm(
        pixel_count,
        generator=generator,
    )


def template_arguments(
    analysis_config: dict,
    seed: int,
) -> dict:
    template_config = analysis_config["template_reuse"]

    return {
        "patch_size": int(
            template_config["patch_size"]
        ),
        "template_count": int(
            template_config["templates"]
        ),
        "folds": int(template_config["folds"]),
        "repetitions": int(
            template_config["repetitions"]
        ),
        "initializations": int(
            template_config["initializations"]
        ),
        "maximum_iterations": int(
            template_config["maximum_iterations"]
        ),
        "tolerance": float(
            template_config["tolerance"]
        ),
        "seed": int(seed),
    }


def analyze_weight_set(
    weights: torch.Tensor,
    metadata: dict,
    analysis_config: dict,
    template_seed: int,
) -> tuple[dict, list[dict]]:
    window_sizes = [
        int(value)
        for value in analysis_config["locality"][
            "window_sizes"
        ]
    ]

    locality_scores = {}

    for window_size in window_sizes:
        scores, _ = maximum_window_energy_fraction(
            weights,
            window_size,
        )
        locality_scores[window_size] = scores

    reuse = cross_validated_template_reuse(
        weights=weights,
        **template_arguments(
            analysis_config,
            template_seed,
        ),
    )

    run_row = dict(metadata)

    for window_size, scores in locality_scores.items():
        run_row[
            f"mean_locality_window_{window_size}"
        ] = float(scores.mean().item())
        run_row[
            f"std_locality_window_{window_size}"
        ] = float(
            scores.std(unbiased=False).item()
        )

    reuse_columns = {
        "conditional_shape_score": (
            "template_conditional_score"
        ),
        "total_energy_reconstruction_fraction": (
            "template_total_reconstruction"
        ),
        "selected_patch_energy_fraction": (
            "template_selected_patch_energy"
        ),
        "maximum_patch_energy_fraction": (
            "maximum_patch_energy_fraction"
        ),
    }

    for source_name, output_name in reuse_columns.items():
        values = reuse[source_name]
        run_row[f"mean_{output_name}"] = float(
            values.mean().item()
        )
        run_row[f"std_{output_name}"] = float(
            values.std(unbiased=False).item()
        )

    neuron_rows = []

    for neuron_index in range(weights.shape[0]):
        row = dict(metadata)
        row["neuron_index"] = neuron_index

        for window_size, scores in locality_scores.items():
            row[f"locality_window_{window_size}"] = float(
                scores[neuron_index].item()
            )

        for source_name, output_name in reuse_columns.items():
            row[output_name] = float(
                reuse[source_name][neuron_index].item()
            )

        neuron_rows.append(row)

    return run_row, neuron_rows

def partial_is_complete(
    run_path: Path,
    neuron_path: Path,
    expected_neurons: int,
) -> bool:
    if not run_path.exists() or not neuron_path.exists():
        return False

    try:
        run_rows = pd.read_csv(run_path)
        neuron_rows = pd.read_csv(neuron_path)
    except Exception:
        return False

    return (
        len(run_rows) == 1
        and len(neuron_rows) == expected_neurons
    )


def analyze_and_save_partial(
    weights: torch.Tensor,
    metadata: dict,
    analysis_config: dict,
    template_seed: int,
    partial_directory: Path,
) -> None:
    analysis_id = metadata["analysis_id"]
    run_path = partial_directory / (
        f"{analysis_id}_run.csv"
    )
    neuron_path = partial_directory / (
        f"{analysis_id}_neurons.csv"
    )

    if partial_is_complete(
        run_path,
        neuron_path,
        weights.shape[0],
    ):
        print(
            f"Skipping completed analysis: {analysis_id}",
            flush=True,
        )
        return

    print(f"Analyzing: {analysis_id}", flush=True)

    run_row, neuron_rows = analyze_weight_set(
        weights=weights,
        metadata=metadata,
        analysis_config=analysis_config,
        template_seed=template_seed,
    )

    run_temporary = run_path.with_suffix(".csv.tmp")
    neuron_temporary = neuron_path.with_suffix(
        ".csv.tmp"
    )

    pd.DataFrame(neuron_rows).to_csv(
        neuron_temporary,
        index=False,
    )
    pd.DataFrame([run_row]).to_csv(
        run_temporary,
        index=False,
    )

    neuron_temporary.replace(neuron_path)
    run_temporary.replace(run_path)

    print(f"Completed: {analysis_id}", flush=True)


def combine_partial_results(
    partial_directory: Path,
    output_directory: Path,
    seed: int,
    expected_analyses: int,
    expected_neurons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_paths = sorted(
        partial_directory.glob("*_run.csv")
    )
    neuron_paths = sorted(
        partial_directory.glob("*_neurons.csv")
    )

    if len(run_paths) != expected_analyses:
        raise ValueError(
            f"Expected {expected_analyses} run partials, "
            f"found {len(run_paths)}."
        )

    if len(neuron_paths) != expected_analyses:
        raise ValueError(
            f"Expected {expected_analyses} neuron partials, "
            f"found {len(neuron_paths)}."
        )

    run_table = pd.concat(
        [pd.read_csv(path) for path in run_paths],
        ignore_index=True,
    )
    neuron_table = pd.concat(
        [pd.read_csv(path) for path in neuron_paths],
        ignore_index=True,
    )

    if len(neuron_table) != (
        expected_analyses * expected_neurons
    ):
        raise ValueError(
            "Combined neuron table has an invalid size."
        )

    run_table.to_csv(
        output_directory
        / f"structural_run_metrics_seed{seed}.csv",
        index=False,
    )
    neuron_table.to_csv(
        output_directory
        / f"structural_neuron_metrics_seed{seed}.csv",
        index=False,
    )

    return run_table, neuron_table

def validate_run(
    run: dict,
    condition: str,
    seed: int,
    expected_permutation_seed: int | None,
) -> None:
    if run["condition"] != condition:
        raise ValueError("Run condition does not match.")
    if int(run["seed"]) != seed:
        raise ValueError("Run seed does not match.")

    actual_permutation_seed = run.get(
        "pixel_permutation_seed"
    )

    if actual_permutation_seed != expected_permutation_seed:
        raise ValueError(
            "Pixel-permutation seed does not match."
        )


def main() -> None:
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

    pixel_config = control_config["pixel_permutation"]
    conditions = list(pixel_config["conditions"])
    final_seeds = {
        int(value)
        for value in pixel_config["final_seeds"]
    }

    if args.seed not in final_seeds:
        raise ValueError("Seed is not a final seed.")

    image_shape = tuple(
        int(value)
        for value in analysis_config["first_layer"][
            "image_shape"
        ]
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
    shuffle_seed = int(shuffle_config["seed"])
    template_base_seed = int(
        analysis_config["template_reuse"]["seed"]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    partial_directory = (
        args.output_directory
        / "partials"
        / f"seed{args.seed}"
    )
    partial_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_neurons = int(
        experiment_config["mlp"]["hidden_width"]
    )
    expected_analyses = 1 + len(conditions) * (
        1 + shuffle_repetitions + 2
    )

    print(f"Device: {device}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
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
            run=primary_run,
            condition=condition,
            seed=args.seed,
            expected_permutation_seed=None,
        )

        primary_weights = primary_weights.to(device)

        primary_metadata = {
            "analysis_id": (
                f"trained_{condition}_seed{args.seed}"
            ),
            "source": "trained",
            "coordinate_system": "original_image",
            "condition": condition,
            "seed": args.seed,
            "run_name": primary_run_name,
            "parent_run_name": "",
            "shuffle_index": -1,
        }

        analyze_and_save_partial(
            weights=primary_weights,
            metadata=primary_metadata,
            analysis_config=analysis_config,
            template_seed=(
                template_base_seed
                + args.seed * 1_000_000
                + condition_index * 10_000
                + 1
            ),
            partial_directory=partial_directory,
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

            shuffled_weights = shuffle_spatial_weights(
                primary_weights,
                generator,
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

            analyze_and_save_partial(
                weights=shuffled_weights,
                metadata=shuffle_metadata,
                analysis_config=analysis_config,
                template_seed=(
                    template_base_seed
                    + args.seed * 1_000_000
                    + condition_index * 10_000
                    + 100
                    + shuffle_index
                ),
                partial_directory=partial_directory,
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
            run=pixel_run,
            condition=condition,
            seed=args.seed,
            expected_permutation_seed=(
                permutation_seed
            ),
        )

        pixel_weights = pixel_weights.to(device)

        model_coordinate_metadata = {
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

        analyze_and_save_partial(
            weights=pixel_weights,
            metadata=model_coordinate_metadata,
            analysis_config=analysis_config,
            template_seed=(
                template_base_seed
                + args.seed * 1_000_000
                + condition_index * 10_000
                + 500
            ),
            partial_directory=partial_directory,
        )

        pixel_count = image_shape[1] * image_shape[2]
        permutation = build_spatial_permutation(
            permutation_seed,
            pixel_count,
        )
        image_coordinate_weights = (
            weights_in_original_image_coordinates(
                pixel_weights,
                permutation,
            )
        )

        image_coordinate_metadata = {
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

        analyze_and_save_partial(
            weights=image_coordinate_weights,
            metadata=image_coordinate_metadata,
            analysis_config=analysis_config,
            template_seed=(
                template_base_seed
                + args.seed * 1_000_000
                + condition_index * 10_000
                + 600
            ),
            partial_directory=partial_directory,
        )

    random_weights = random_initialization_weights(
        experiment_config=experiment_config,
        seed=args.seed,
        image_shape=image_shape,
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

    analyze_and_save_partial(
        weights=random_weights,
        metadata=random_metadata,
        analysis_config=analysis_config,
        template_seed=(
            template_base_seed
            + args.seed * 1_000_000
            + 900_000
        ),
        partial_directory=partial_directory,
    )

    run_table, _ = combine_partial_results(
        partial_directory=partial_directory,
        output_directory=args.output_directory,
        seed=args.seed,
        expected_analyses=expected_analyses,
        expected_neurons=expected_neurons,
    )

    display_columns = [
        "source",
        "coordinate_system",
        "condition",
        "mean_locality_window_7",
        "mean_template_total_reconstruction",
    ]

    display = run_table[
        run_table["source"] != "spatial_shuffle"
    ][display_columns].sort_values(
        [
            "source",
            "coordinate_system",
            "condition",
        ]
    )

    print(display.round(6).to_string(index=False))
    print(
        f"Results saved to: {args.output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
