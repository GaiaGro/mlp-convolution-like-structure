import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data import SplitSpec, build_cifar10_datasets
from src.data.transforms import translate_batch
from src.evaluation.translation import (
    PRIMARY_PADDING_MODES,
    build_axial_shifts,
)
from src.models import build_model
from src.utils import get_device


EXPECTED_SEEDS = {43, 44, 45, 46, 47}
EXPECTED_CONDITIONS = {
    "mlp": {
        "baseline",
        "exposure",
        "consistency",
        "augmentation",
        "augmentation_consistency",
    },
    "cnn": {"baseline", "augmentation"},
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate all final checkpoints on CIFAR-10."
    )
    parser.add_argument(
        "--training-directory",
        type=Path,
        default=Path("results/final/training"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/final/evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def normalized_auc(distances, values):
    pairs = sorted(zip(distances, values))
    if len(pairs) < 2:
        raise ValueError("At least two distances are required.")

    area = sum(
        (right_distance - left_distance)
        * (left_value + right_value)
        / 2
        for (
            left_distance,
            left_value,
        ), (
            right_distance,
            right_value,
        ) in zip(pairs[:-1], pairs[1:])
    )
    return area / (pairs[-1][0] - pairs[0][0])


def prediction_key(padding_mode, dx, dy):
    dx_label = str(dx).replace("-", "m")
    dy_label = str(dy).replace("-", "m")
    return f"{padding_mode}_dx_{dx_label}_dy_{dy_label}"


@torch.inference_mode()
def evaluate_checkpoint(
    model,
    loader,
    device,
    run,
    shifts,
    padding_modes,
    prediction_path,
):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    combinations = [
        (padding_mode, shift)
        for padding_mode in padding_modes
        for shift in shifts
    ]
    samples = 0
    clean_correct = 0
    clean_loss = 0.0
    shifted_correct = {
        combination: 0 for combination in combinations
    }
    consistent = {
        combination: 0 for combination in combinations
    }

    target_parts = []
    clean_prediction_parts = []
    shifted_prediction_parts = {
        combination: [] for combination in combinations
    }

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        clean_logits = model(images)
        clean_predictions = clean_logits.argmax(dim=1)

        batch_size = targets.shape[0]
        samples += batch_size
        clean_loss += criterion(clean_logits, targets).item()
        clean_correct += (
            clean_predictions == targets
        ).sum().item()

        target_parts.append(
            targets.cpu().numpy().astype(np.int16)
        )
        clean_prediction_parts.append(
            clean_predictions.cpu().numpy().astype(np.int16)
        )

        for combination in combinations:
            padding_mode, (dx, dy) = combination
            translated = translate_batch(
                images,
                dx=dx,
                dy=dy,
                padding_mode=padding_mode,
            )
            predictions = model(translated).argmax(dim=1)

            shifted_correct[combination] += (
                predictions == targets
            ).sum().item()
            consistent[combination] += (
                predictions == clean_predictions
            ).sum().item()
            shifted_prediction_parts[combination].append(
                predictions.cpu().numpy().astype(np.int16)
            )

    if samples == 0:
        raise ValueError("Test loader is empty.")

    clean_accuracy = clean_correct / samples
    shift_rows = []

    for padding_mode, (dx, dy) in combinations:
        shift_rows.append(
            {
                "run_name": run["run_name"],
                "model": run["model"],
                "condition": run["condition"],
                "seed": int(run["seed"]),
                "padding_mode": padding_mode,
                "distance": abs(dx) + abs(dy),
                "dx": dx,
                "dy": dy,
                "clean_accuracy": clean_accuracy,
                "shifted_accuracy": (
                    shifted_correct[(padding_mode, (dx, dy))]
                    / samples
                ),
                "prediction_consistency": (
                    consistent[(padding_mode, (dx, dy))]
                    / samples
                ),
            }
        )

    shift_table = pd.DataFrame(shift_rows)
    mode_accuracy = (
        shift_table.groupby("padding_mode")[
            "shifted_accuracy"
        ].mean()
    )
    mode_consistency = (
        shift_table.groupby("padding_mode")[
            "prediction_consistency"
        ].mean()
    )
    distance_accuracy = (
        shift_table.groupby("distance")[
            "shifted_accuracy"
        ].mean()
    )

    mode_aucs = {}
    for padding_mode, table in shift_table.groupby(
        "padding_mode"
    ):
        by_distance = table.groupby("distance")[
            "shifted_accuracy"
        ].mean()
        mode_aucs[padding_mode] = normalized_auc(
            by_distance.index.tolist(),
            by_distance.tolist(),
        )

    distance_four = shift_table[
        shift_table["distance"] == 4
    ]
    distance_four_by_mode = (
        distance_four.groupby("padding_mode")[
            "shifted_accuracy"
        ].mean()
    )

    run_row = {
        "run_name": run["run_name"],
        "model": run["model"],
        "condition": run["condition"],
        "seed": int(run["seed"]),
        "checkpoint_epoch": int(run["checkpoint_epoch"]),
        "test_loss": clean_loss / samples,
        "clean_accuracy": clean_accuracy,
        "mean_shifted_accuracy": float(
            shift_table["shifted_accuracy"].mean()
        ),
        "worst_padding_shifted_accuracy": float(
            mode_accuracy.min()
        ),
        "mean_prediction_consistency": float(
            shift_table["prediction_consistency"].mean()
        ),
        "worst_padding_prediction_consistency": float(
            mode_consistency.min()
        ),
        "distance_accuracy_auc": normalized_auc(
            distance_accuracy.index.tolist(),
            distance_accuracy.tolist(),
        ),
        "worst_padding_distance_auc": min(
            mode_aucs.values()
        ),
        "heldout_distance4_mean_accuracy": float(
            distance_four["shifted_accuracy"].mean()
        ),
        "heldout_distance4_worst_padding_accuracy": float(
            distance_four_by_mode.min()
        ),
    }

    for distance, value in distance_accuracy.items():
        run_row[f"distance{int(distance)}_mean_accuracy"] = (
            float(value)
        )

    arrays = {
        "targets": np.concatenate(target_parts),
        "clean_predictions": np.concatenate(
            clean_prediction_parts
        ),
    }
    for padding_mode, (dx, dy) in combinations:
        arrays[prediction_key(padding_mode, dx, dy)] = (
            np.concatenate(
                shifted_prediction_parts[
                    (padding_mode, (dx, dy))
                ]
            )
        )

    np.savez_compressed(prediction_path, **arrays)
    return run_row, shift_rows


def main():
    args = parse_arguments()

    config = yaml.safe_load(
        Path("configs/experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataset_config = config["dataset"]
    distances = config["translation"]["test_distances"]
    shifts = build_axial_shifts(distances)

    datasets, _ = build_cifar10_datasets(
        root=dataset_config["root"],
        spec=SplitSpec(
            train_size=dataset_config["train_size"],
            validation_size=dataset_config[
                "validation_size"
            ],
            seed=dataset_config["split_seed"],
        ),
        use_augmentation=False,
        download=False,
    )

    device = get_device()
    loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    config_paths = sorted(
        (args.training_directory / "configs").glob("*.yaml")
    )
    if len(config_paths) != 35:
        raise ValueError(
            f"Expected 35 final configurations, found "
            f"{len(config_paths)}."
        )

    runs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in config_paths
    ]
    observed = {
        (
            run["model"],
            run["condition"],
            int(run["seed"]),
        )
        for run in runs
    }
    expected = {
        (model, condition, seed)
        for model, conditions in EXPECTED_CONDITIONS.items()
        for condition in conditions
        for seed in EXPECTED_SEEDS
    }
    if observed != expected:
        raise ValueError(
            "Final configurations do not match the "
            "predefined model-condition-seed design."
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    prediction_directory = (
        args.output_directory / "predictions"
    )
    prediction_directory.mkdir(parents=True, exist_ok=True)

    run_rows = []
    shift_rows = []

    for run in sorted(
        runs,
        key=lambda item: (
            item["model"],
            item["condition"],
            int(item["seed"]),
        ),
    ):
        run_name = run["run_name"]
        checkpoint_path = (
            args.training_directory
            / "checkpoints"
            / f"{run_name}_best.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model = build_model(run["model"], config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        evaluation_run = dict(run)
        evaluation_run["checkpoint_epoch"] = checkpoint[
            "epoch"
        ]

        print(
            f"Evaluating {run_name} "
            f"at epoch {checkpoint['epoch']}",
            flush=True,
        )

        run_row, current_shift_rows = evaluate_checkpoint(
            model=model,
            loader=loader,
            device=device,
            run=evaluation_run,
            shifts=shifts,
            padding_modes=PRIMARY_PADDING_MODES,
            prediction_path=(
                prediction_directory
                / f"{run_name}.npz"
            ),
        )
        run_rows.append(run_row)
        shift_rows.extend(current_shift_rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_table = pd.DataFrame(run_rows).sort_values(
        ["model", "condition", "seed"]
    )
    shift_table = pd.DataFrame(shift_rows).sort_values(
        [
            "model",
            "condition",
            "seed",
            "padding_mode",
            "distance",
            "dx",
            "dy",
        ]
    )
    distance_table = (
        shift_table.groupby(
            [
                "run_name",
                "model",
                "condition",
                "seed",
                "padding_mode",
                "distance",
            ],
            as_index=False,
        )
        .agg(
            shifted_accuracy=("shifted_accuracy", "mean"),
            prediction_consistency=(
                "prediction_consistency",
                "mean",
            ),
        )
        .sort_values(
            [
                "model",
                "condition",
                "seed",
                "padding_mode",
                "distance",
            ]
        )
    )

    run_table.to_csv(
        args.output_directory / "test_runs.csv",
        index=False,
    )
    shift_table.to_csv(
        args.output_directory / "test_shifts.csv",
        index=False,
    )
    distance_table.to_csv(
        args.output_directory / "test_distances.csv",
        index=False,
    )

    print(run_table.round(6).to_string(index=False))
    print(f"Results saved to: {args.output_directory}")


if __name__ == "__main__":
    main()
