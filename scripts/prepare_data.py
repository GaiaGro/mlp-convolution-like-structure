from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.cifar10 import (
    SplitSpec,
    build_cifar10_datasets,
    write_split_manifest,
)


def main() -> None:
    config = yaml.safe_load(
        Path("configs/experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataset_config = config["dataset"]

    specification = SplitSpec(
        train_size=dataset_config["train_size"],
        validation_size=dataset_config["validation_size"],
        seed=dataset_config["split_seed"],
    )

    datasets, indices = build_cifar10_datasets(
        root=dataset_config["root"],
        spec=specification,
        use_augmentation=False,
        download=True,
    )

    output = Path("results/splits")
    write_split_manifest(
        indices,
        output / "cifar10_indices.npz",
    )

    targets = np.asarray(datasets["train"].dataset.targets)
    records = []

    for split, split_indices in indices.items():
        counts = np.bincount(
            targets[split_indices],
            minlength=10,
        )
        for class_index, count in enumerate(counts):
            records.append(
                {
                    "split": split,
                    "class_index": class_index,
                    "count": int(count),
                }
            )

    pd.DataFrame(records).to_csv(
        output / "cifar10_split_summary.csv",
        index=False,
    )

    print("CIFAR-10 prepared.")
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)}")


if __name__ == "__main__":
    main()
