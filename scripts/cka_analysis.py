from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import SplitSpec, build_cifar10_datasets
from src.models import build_model
from src.utils import get_device


TRAINING_DIR = Path("results/final/training")
OUTPUT_DIR = Path("results/cka")
CONFIG_PATH = Path("configs/experiment.yaml")

SEEDS = [43, 44, 45, 46, 47]

MLP_CONDITIONS = [
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
]

CNN_REFERENCES = [
    "baseline",
    "augmentation",
]

MLP_LAYER_LABELS = [
    "Projection",
    "Block 1",
    "Block 2",
    "Block 3",
    "Output norm",
]

CNN_LAYER_LABELS = [
    "Stem",
    "Stage 1",
    "Stage 2",
    "Stage 3a",
    "Stage 3b",
    "Stage 4",
    "Global pool",
]


def load_config():
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def load_model(
    model_name,
    condition,
    seed,
    config,
    device,
):
    run_name = (
        f"{model_name}_{condition}_seed{seed}"
    )

    path = (
        TRAINING_DIR
        / "checkpoints"
        / f"{run_name}_best.pt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        model_name,
        config,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def representation_modules(
    model,
    model_name,
):
    if model_name == "mlp":
        modules = [
            (
                "Projection",
                model.input_activation,
            ),
            (
                "Block 1",
                model.blocks[0],
            ),
            (
                "Block 2",
                model.blocks[1],
            ),
            (
                "Block 3",
                model.blocks[2],
            ),
            (
                "Output norm",
                model.output_norm,
            ),
        ]

        return modules

    if model_name == "cnn":
        if len(model.stages) != 5:
            raise RuntimeError(
                "Expected five CNN residual blocks."
            )

        modules = [
            (
                "Stem",
                model.stem,
            ),
            (
                "Stage 1",
                model.stages[0],
            ),
            (
                "Stage 2",
                model.stages[1],
            ),
            (
                "Stage 3a",
                model.stages[2],
            ),
            (
                "Stage 3b",
                model.stages[3],
            ),
            (
                "Stage 4",
                model.stages[4],
            ),
            (
                "Global pool",
                model.pool,
            ),
        ]

        return modules

    raise ValueError(model_name)


def vectorize_representation(output):
    if isinstance(output, tuple):
        output = output[0]

    if output.ndim == 4:
        # Global average pooling over spatial
        # coordinates gives one representation
        # vector per image.
        return output.mean(dim=(2, 3))

    if output.ndim == 2:
        return output

    return output.flatten(start_dim=1)


@torch.inference_mode()
def collect_representations(
    model,
    model_name,
    loader,
    device,
):
    modules = representation_modules(
        model,
        model_name,
    )

    parts = {
        name: []
        for name, _ in modules
    }

    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            values = vectorize_representation(
                output
            )

            parts[name].append(
                values.detach().cpu()
            )

        return hook

    for name, module in modules:
        handles.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )

    try:
        for images, _targets in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            model(images)

    finally:
        for handle in handles:
            handle.remove()

    representations = {}

    for name, values in parts.items():
        matrix = torch.cat(
            values,
            dim=0,
        ).float()

        if matrix.ndim != 2:
            raise RuntimeError(
                f"{name}: expected matrix, "
                f"found {matrix.shape}"
            )

        if not torch.isfinite(matrix).all():
            raise RuntimeError(
                f"{name}: non-finite activations."
            )

        representations[name] = matrix

    return representations


def center_features(matrix):
    return (
        matrix
        - matrix.mean(
            dim=0,
            keepdim=True,
        )
    )


def linear_cka(x, y):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "CKA expects two matrices."
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "CKA representations must use "
            "the same examples."
        )

    x = center_features(x)
    y = center_features(y)

    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y

    numerator = (
        cross.double()
        .square()
        .sum()
    )

    x_norm = (
        xx.double()
        .square()
        .sum()
        .sqrt()
    )

    y_norm = (
        yy.double()
        .square()
        .sum()
        .sqrt()
    )

    denominator = x_norm * y_norm

    if denominator <= 0:
        raise RuntimeError(
            "Degenerate representation in CKA."
        )

    value = (
        numerator
        / denominator
    ).item()

    if value < -1e-6 or value > 1.0001:
        raise RuntimeError(
            f"Unexpected CKA value: {value}"
        )

    return float(
        min(
            max(value, 0.0),
            1.0,
        )
    )


def validate_cka():
    generator = torch.Generator()
    generator.manual_seed(2026)

    x = torch.randn(
        200,
        40,
        generator=generator,
    )

    same = linear_cka(
        x,
        x,
    )

    scaled = linear_cka(
        x,
        7.3 * x,
    )

    permutation = torch.randperm(
        x.shape[1],
        generator=generator,
    )

    permuted = linear_cka(
        x,
        x[:, permutation],
    )

    for name, value in [
        ("self", same),
        ("scale", scaled),
        ("permutation", permuted),
    ]:
        if abs(value - 1.0) > 1e-5:
            raise RuntimeError(
                f"CKA validation failed: "
                f"{name}={value}"
            )


def summarize(rows):
    values = pd.DataFrame(rows)

    group_columns = [
        "cnn_condition",
        "mlp_condition",
        "mlp_layer",
        "cnn_layer",
    ]

    summary_rows = []

    # t critical value for 95% CI with df=4.
    t_critical = 2.7764451051977987

    for keys, group in values.groupby(
        group_columns,
        sort=False,
    ):
        array = group[
            "cka"
        ].to_numpy(dtype=float)

        if len(array) != 5:
            raise RuntimeError(
                f"Expected 5 seeds for {keys}, "
                f"found {len(array)}."
            )

        mean = float(array.mean())
        std = float(
            array.std(ddof=1)
        )

        half_width = (
            t_critical
            * std
            / np.sqrt(len(array))
        )

        summary_rows.append(
            {
                "cnn_condition": keys[0],
                "mlp_condition": keys[1],
                "mlp_layer": keys[2],
                "cnn_layer": keys[3],
                "n_seeds": len(array),
                "mean_cka": mean,
                "std_cka": std,
                "ci95_lower": (
                    mean - half_width
                ),
                "ci95_upper": (
                    mean + half_width
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def build_best_match_table(summary):
    rows = []

    for cnn_condition in CNN_REFERENCES:
        for mlp_condition in MLP_CONDITIONS:
            for mlp_layer in MLP_LAYER_LABELS:
                subset = summary[
                    (
                        summary[
                            "cnn_condition"
                        ]
                        == cnn_condition
                    )
                    & (
                        summary[
                            "mlp_condition"
                        ]
                        == mlp_condition
                    )
                    & (
                        summary[
                            "mlp_layer"
                        ]
                        == mlp_layer
                    )
                ].copy()

                if len(subset) != len(
                    CNN_LAYER_LABELS
                ):
                    raise RuntimeError(
                        "Incomplete CKA heatmap."
                    )

                best = subset.loc[
                    subset[
                        "mean_cka"
                    ].idxmax()
                ]

                rows.append(
                    {
                        "cnn_condition": (
                            cnn_condition
                        ),
                        "mlp_condition": (
                            mlp_condition
                        ),
                        "mlp_layer": (
                            mlp_layer
                        ),
                        "best_cnn_layer": (
                            best["cnn_layer"]
                        ),
                        "mean_cka": float(
                            best["mean_cka"]
                        ),
                        "ci95_lower": float(
                            best[
                                "ci95_lower"
                            ]
                        ),
                        "ci95_upper": float(
                            best[
                                "ci95_upper"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main():
    validate_cka()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config()

    dataset_config = config[
        "dataset"
    ]

    datasets, _ = (
        build_cifar10_datasets(
            root=dataset_config["root"],
            spec=SplitSpec(
                train_size=dataset_config[
                    "train_size"
                ],
                validation_size=(
                    dataset_config[
                        "validation_size"
                    ]
                ),
                seed=dataset_config[
                    "split_seed"
                ],
            ),
            use_augmentation=False,
            download=False,
        )
    )

    device = get_device()

    loader = DataLoader(
        datasets["test"],
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=True,
    )

    if len(datasets["test"]) != 10000:
        raise RuntimeError(
            "Expected 10,000 CIFAR-10 "
            "test examples."
        )

    rows = []

    for seed in SEEDS:
        print(
            f"\n=== Seed {seed} ===",
            flush=True,
        )

        cnn_representations = {}

        for cnn_condition in CNN_REFERENCES:
            print(
                f"Collecting CNN "
                f"{cnn_condition}...",
                flush=True,
            )

            cnn_model = load_model(
                "cnn",
                cnn_condition,
                seed,
                config,
                device,
            )

            cnn_representations[
                cnn_condition
            ] = collect_representations(
                cnn_model,
                "cnn",
                loader,
                device,
            )

            del cnn_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

        for mlp_condition in MLP_CONDITIONS:
            print(
                f"Collecting MLP "
                f"{mlp_condition}...",
                flush=True,
            )

            mlp_model = load_model(
                "mlp",
                mlp_condition,
                seed,
                config,
                device,
            )

            mlp_representations = (
                collect_representations(
                    mlp_model,
                    "mlp",
                    loader,
                    device,
                )
            )

            del mlp_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

            for cnn_condition in CNN_REFERENCES:
                cnn_features = (
                    cnn_representations[
                        cnn_condition
                    ]
                )

                print(
                    f"  CKA vs CNN "
                    f"{cnn_condition}",
                    flush=True,
                )

                for mlp_layer in MLP_LAYER_LABELS:
                    x = (
                        mlp_representations[
                            mlp_layer
                        ]
                    )

                    for cnn_layer in CNN_LAYER_LABELS:
                        y = cnn_features[
                            cnn_layer
                        ]

                        value = linear_cka(
                            x,
                            y,
                        )

                        rows.append(
                            {
                                "seed": seed,
                                "mlp_condition": (
                                    mlp_condition
                                ),
                                "cnn_condition": (
                                    cnn_condition
                                ),
                                "mlp_layer": (
                                    mlp_layer
                                ),
                                "cnn_layer": (
                                    cnn_layer
                                ),
                                "mlp_dimension": (
                                    x.shape[1]
                                ),
                                "cnn_dimension": (
                                    y.shape[1]
                                ),
                                "n_examples": (
                                    x.shape[0]
                                ),
                                "cka": value,
                            }
                        )

            del mlp_representations

        del cnn_representations

    values = pd.DataFrame(rows)

    expected_rows = (
        len(SEEDS)
        * len(MLP_CONDITIONS)
        * len(CNN_REFERENCES)
        * len(MLP_LAYER_LABELS)
        * len(CNN_LAYER_LABELS)
    )

    if len(values) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} CKA "
            f"values, found {len(values)}."
        )

    values.to_csv(
        OUTPUT_DIR
        / "cka_paired_seed_values.csv",
        index=False,
    )

    summary = summarize(rows)

    summary.to_csv(
        OUTPUT_DIR
        / "cka_paired_summary.csv",
        index=False,
    )

    best_matches = (
        build_best_match_table(
            summary
        )
    )

    best_matches.to_csv(
        OUTPUT_DIR
        / "cka_best_matching_layers.csv",
        index=False,
    )

    print()
    print("CKA analysis complete.")
    print(
        f"Pairwise rows: {len(values)}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
