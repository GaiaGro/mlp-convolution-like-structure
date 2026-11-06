import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data import SplitSpec, build_cifar10_datasets
from src.evaluation import (
    build_axial_shifts,
    evaluate_translation_robustness,
)
from src.models import build_model, count_parameters
from src.training.engine import (
    create_scheduler,
    train_one_epoch,
)
from src.training.objectives import (
    AUGMENTED_CONDITIONS,
    CONDITIONS,
)
from src.utils import get_device, set_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one predefined CIFAR-10 experiment."
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn"],
        required=True,
    )
    parser.add_argument(
        "--condition",
        choices=sorted(CONDITIONS),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--variance-covariance-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pixel-permutation-seed",
        type=int,
        help=(
            "Fixed spatial-pixel permutation seed for the "
            "structural control."
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--run-name",
        help="Optional unique name for the run.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/training"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.model == "cnn" and args.condition not in {
        "baseline",
        "augmentation",
    }:
        raise ValueError(
            "CNN is only defined for baseline and augmentation."
        )

    if (
        args.pixel_permutation_seed is not None
        and args.model != "mlp"
    ):
        raise ValueError(
            "Pixel permutation is only defined for the MLP."
        )

    if (
        args.pixel_permutation_seed is not None
        and args.pixel_permutation_seed < 0
    ):
        raise ValueError(
            "Pixel permutation seed must be non-negative."
        )

    consistency_conditions = {
        "consistency",
        "augmentation_consistency",
    }

    if (
        args.condition in consistency_conditions
        and args.consistency_weight <= 0
    ):
        raise ValueError(
            "Consistency conditions require a positive "
            "consistency weight."
        )

    if args.variance_covariance_weight < 0:
        raise ValueError(
            "Variance-covariance weight must be non-negative."
        )

    if (
        args.condition not in consistency_conditions
        and (
            args.consistency_weight != 0
            or args.variance_covariance_weight != 0
        )
    ):
        raise ValueError(
            "This condition requires both regularization "
            "weights to be zero."
        )

    set_seed(args.seed)

    config = yaml.safe_load(
        Path("configs/experiment.yaml").read_text(
            encoding="utf-8"
        )
    )

    training_config = config["training"]
    dataset_config = config["dataset"]
    consistency_config = config["consistency"]
    translation_config = config["translation"]

    epochs = args.epochs or training_config["epochs"]
    batch_size = (
        args.batch_size or training_config["batch_size"]
    )
    learning_rate = (
        args.learning_rate
        or training_config["learning_rate_candidates"][0]
    )
    weight_decay = (
        args.weight_decay
        or training_config["weight_decay_candidates"][0]
    )

    training_shifts = tuple(
        tuple(int(value) for value in shift)
        for shift in translation_config["train_shifts"]
    )
    training_padding_modes = tuple(
        translation_config["training_padding_modes"]
    )
    evaluation_padding_modes = tuple(
        translation_config[
            "primary_evaluation_padding_modes"
        ]
    )
    validation_shifts = build_axial_shifts(
        translation_config["validation_distances"]
    )

    if (
        translation_config["validation_aggregation"]
        != "worst_padding"
    ):
        raise ValueError(
            "Only worst_padding validation is supported."
        )

    use_augmentation = (
        args.condition in AUGMENTED_CONDITIONS
    )

    datasets, _ = build_cifar10_datasets(
        root=dataset_config["root"],
        spec=SplitSpec(
            train_size=dataset_config["train_size"],
            validation_size=(
                dataset_config["validation_size"]
            ),
            seed=dataset_config["split_seed"],
        ),
        use_augmentation=use_augmentation,
        download=False,
    )

    if args.max_train_samples is not None:
        datasets["train"] = Subset(
            datasets["train"],
            range(
                min(
                    args.max_train_samples,
                    len(datasets["train"]),
                )
            ),
        )

    if args.max_validation_samples is not None:
        datasets["validation"] = Subset(
            datasets["validation"],
            range(
                min(
                    args.max_validation_samples,
                    len(datasets["validation"]),
                )
            ),
        )

    device = get_device()
    pin_memory = device.type == "cuda"

    loader_generator = torch.Generator().manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
    )

    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        args.model,
        config,
        pixel_permutation_seed=args.pixel_permutation_seed,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=training_config["label_smoothing"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    warmup_epochs = min(
        training_config["warmup_epochs"],
        max(0, epochs - 1),
    )
    scheduler = create_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
    )

    mixed_precision = (
        training_config["mixed_precision"]
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision,
    )

    weight_label = str(args.consistency_weight).replace(
        ".",
        "p",
    )
    rho_label = str(
        args.variance_covariance_weight
    ).replace(".", "p")
    default_run_name = (
        f"{args.model}_{args.condition}_"
        f"lambda{weight_label}_rho{rho_label}_"
        f"seed{args.seed}"
    )
    run_name = args.run_name or default_run_name

    checkpoint_directory = (
        args.output_directory / "checkpoints"
    )
    history_directory = args.output_directory / "histories"
    config_directory = args.output_directory / "configs"

    for directory in (
        checkpoint_directory,
        history_directory,
        config_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "model": args.model,
        "condition": args.condition,
        "seed": args.seed,
        "consistency_weight": args.consistency_weight,
        "variance_covariance_weight": (
            args.variance_covariance_weight
        ),
        "pixel_permutation_seed": (
            args.pixel_permutation_seed
        ),
        "variance_target": consistency_config[
            "variance_target"
        ],
        "variance_epsilon": consistency_config[
            "variance_epsilon"
        ],
        "covariance_coefficient": consistency_config[
            "covariance_coefficient"
        ],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "experiment_config": config,
    }

    (
        config_directory / f"{run_name}.yaml"
    ).write_text(
        yaml.safe_dump(run_config, sort_keys=False),
        encoding="utf-8",
    )

    history = []
    best_score = float("-inf")
    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Run: {run_name}")
    print(f"Parameters: {count_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        translation_generator = (
            torch.Generator().manual_seed(
                args.seed * 100000 + epoch
            )
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            condition=args.condition,
            consistency_weight=args.consistency_weight,
            variance_covariance_weight=(
                args.variance_covariance_weight
            ),
            variance_target=consistency_config[
                "variance_target"
            ],
            epsilon=consistency_config[
                "variance_epsilon"
            ],
            covariance_coefficient=consistency_config[
                "covariance_coefficient"
            ],
            device=device,
            scaler=scaler,
            mixed_precision=mixed_precision,
            translation_generator=translation_generator,
            paired_stochastic_masks=consistency_config[
                "paired_stochastic_masks"
            ],
            translation_shifts=training_shifts,
            translation_padding_modes=training_padding_modes,
        )

        should_validate = (
            epoch % args.validation_interval == 0
            or epoch == epochs
        )

        if should_validate:
            validation_metrics = (
                evaluate_translation_robustness(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    shifts=validation_shifts,
                    padding_modes=evaluation_padding_modes,
                )
            )

            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **train_metrics,
                **validation_metrics,
            }
            history.append(row)

            score = validation_metrics[
                "validation_score"
            ]
            loss = validation_metrics[
                "validation_loss"
            ]

            is_better = (
                score > best_score
                or (
                    score == best_score
                    and loss < best_loss
                )
            )

            if is_better:
                best_score = score
                best_loss = loss
                torch.save(
                    {
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "epoch": epoch,
                        "validation_metrics": (
                            validation_metrics
                        ),
                        "run_config": run_config,
                    },
                    checkpoint_directory
                    / f"{run_name}_best.pt",
                )

            print(
                f"Epoch {epoch:03d} | "
                f"train {train_metrics['train_loss']:.4f} | "
                f"clean {validation_metrics['clean_accuracy']:.4f} | "
                f"worst shifted "
                f"{validation_metrics['worst_padding_shifted_accuracy']:.4f} | "
                f"score {score:.4f}",
                flush=True,
            )

        scheduler.step()

    pd.DataFrame(history).to_csv(
        history_directory / f"{run_name}.csv",
        index=False,
    )

    print(f"Best validation score: {best_score:.6f}")
    print(f"Results saved to: {args.output_directory}")


if __name__ == "__main__":
    main()
