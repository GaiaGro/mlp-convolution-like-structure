from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from src.data.transforms import build_cifar10_transforms


@dataclass(frozen=True)
class SplitSpec:
    train_size: int = 45000
    validation_size: int = 5000
    seed: int = 2026

    @property
    def total_size(self) -> int:
        return self.train_size + self.validation_size


def stratified_split_indices(
    targets: np.ndarray,
    spec: SplitSpec,
) -> dict[str, np.ndarray]:
    targets = np.asarray(targets)

    if len(targets) != spec.total_size:
        raise ValueError(
            f"Expected {spec.total_size} targets, "
            f"found {len(targets)}."
        )

    indices = np.arange(len(targets))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=spec.validation_size,
        random_state=spec.seed,
        shuffle=True,
        stratify=targets,
    )

    result = {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
    }

    combined = np.concatenate(list(result.values()))
    if len(np.unique(combined)) != len(targets):
        raise AssertionError("Splits overlap or omit images.")

    return result


def build_cifar10_datasets(
    root: str | Path,
    spec: SplitSpec,
    use_augmentation: bool,
    download: bool = True,
):
    training_transform, evaluation_transform = (
        build_cifar10_transforms(use_augmentation)
    )

    training_base = CIFAR10(
        root=root,
        train=True,
        transform=training_transform,
        download=download,
    )
    evaluation_base = CIFAR10(
        root=root,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_dataset = CIFAR10(
        root=root,
        train=False,
        transform=evaluation_transform,
        download=download,
    )

    indices = stratified_split_indices(
        np.asarray(training_base.targets),
        spec,
    )

    datasets = {
        "train": Subset(training_base, indices["train"]),
        "validation": Subset(
            evaluation_base,
            indices["validation"],
        ),
        "test": test_dataset,
    }

    return datasets, indices


def write_split_manifest(
    indices: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **indices)
