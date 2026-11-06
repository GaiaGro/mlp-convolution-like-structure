import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import SplitSpec, build_cifar10_datasets
from src.models import build_model
from src.utils import get_device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure representation-collapse diagnostics for "
            "the consistency development grid."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/development/consistency_grid"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=5000)
    return parser.parse_args()


def representation_statistics(
    features: torch.Tensor,
) -> tuple[float, torch.Tensor, float, float]:
    if features.ndim != 2:
        raise ValueError(
            "Expected features with shape (samples, dimensions)."
        )
    if features.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    features = features.to(dtype=torch.float64, device="cpu")
    centered = features - features.mean(dim=0)
    variances = features.var(dim=0, unbiased=True)
    covariance = (
        centered.T @ centered
    ) / (features.shape[0] - 1)

    eigenvalues = torch.linalg.eigvalsh(
        covariance
    ).clamp_min(0)
    eigenvalue_sum = eigenvalues.sum()

    if eigenvalue_sum <= 0:
        effective_rank = 0.0
    else:
        probabilities = (
            eigenvalues[eigenvalues > 0]
            / eigenvalue_sum
        )
        entropy = -(
            probabilities * probabilities.log()
        ).sum()
        effective_rank = float(entropy.exp())

    mean_standard_deviation = float(
        torch.sqrt(variances.clamp_min(0)).mean()
    )
    total_variance = float(variances.sum())

    return (
        effective_rank,
        variances,
        mean_standard_deviation,
        total_variance,
    )


@torch.inference_mode()
def collect_features(
    model,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> torch.Tensor:
    model.eval()
    chunks = []
    collected = 0

    for images, _ in loader:
        images = images.to(device)
        features = model.forward_features(images).detach().cpu()

        remaining = max_samples - collected
        if remaining <= 0:
            break

        features = features[:remaining]
        chunks.append(features)
        collected += features.shape[0]

    if not chunks:
        raise ValueError("No validation features collected.")

    return torch.cat(chunks, dim=0)


def main() -> None:
    args = parse_arguments()

    if args.max_samples < 2:
        raise ValueError("max-samples must be at least two.")

    experiment = yaml.safe_load(
        Path("configs/experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataset_config = experiment["dataset"]
    threshold_fraction = experiment["consistency"][
        "active_dimension_variance_threshold_fraction"
    ]

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
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    config_paths = sorted(
        (args.root / "configs").glob("*.yaml")
    )
    if not config_paths:
        raise FileNotFoundError(
            f"No run configurations found under {args.root}."
        )

    rows = []
    variances_by_run = {}

    print(f"Device: {device}")
    print(f"Runs: {len(config_paths)}")

    for config_path in config_paths:
        run = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        run_name = run["run_name"]
        checkpoint_path = (
            args.root
            / "checkpoints"
            / f"{run_name}_best.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model = build_model(
            run["model"],
            run["experiment_config"],
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model = model.to(device)

        features = collect_features(
            model,
            loader,
            device,
            max_samples=args.max_samples,
        )

        (
            effective_rank,
            variances,
            mean_standard_deviation,
            total_variance,
        ) = representation_statistics(features)

        variances_by_run[run_name] = variances

        rows.append(
            {
                "run_name": run_name,
                "condition": run["condition"],
                "seed": int(run["seed"]),
                "lambda": float(
                    run["consistency_weight"]
                ),
                "rho": float(
                    run["variance_covariance_weight"]
                ),
                "samples": int(features.shape[0]),
                "dimensions": int(features.shape[1]),
                "effective_rank": effective_rank,
                "mean_standard_deviation": (
                    mean_standard_deviation
                ),
                "total_variance": total_variance,
            }
        )

        print(
            f"{run_name} | "
            f"rank {effective_rank:.3f} | "
            f"mean std {mean_standard_deviation:.4f}",
            flush=True,
        )

    table = pd.DataFrame(rows)

    exposure = table[
        table["condition"] == "exposure"
    ].set_index("seed")

    expected_seeds = set(
        experiment["consistency"]["development_seeds"]
    )
    if set(exposure.index) != expected_seeds:
        raise ValueError(
            "Exactly one exposure reference is required "
            "for every development seed."
        )

    active_fractions = []
    rank_retentions = []
    active_retentions = []

    for row in table.itertuples(index=False):
        reference = exposure.loc[row.seed]
        reference_run = reference["run_name"]
        reference_variances = variances_by_run[
            reference_run
        ]

        threshold = (
            threshold_fraction
            * float(reference_variances.mean())
        )

        active_fraction = float(
            (
                variances_by_run[row.run_name]
                >= threshold
            ).double().mean()
        )
        reference_active_fraction = float(
            (
                reference_variances >= threshold
            ).double().mean()
        )

        active_fractions.append(active_fraction)
        rank_retentions.append(
            row.effective_rank
            / float(reference["effective_rank"])
        )
        active_retentions.append(
            active_fraction / reference_active_fraction
        )

    table["active_dimension_fraction"] = active_fractions
    table["effective_rank_retention"] = rank_retentions
    table[
        "active_dimension_fraction_retention"
    ] = active_retentions

    output = args.root / "representation_diagnostics.csv"
    table.to_csv(output, index=False)

    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
