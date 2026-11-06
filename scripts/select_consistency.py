import argparse
import math
from pathlib import Path

import pandas as pd
import yaml


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select lambda and rho from the consistency grid."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/development/consistency_grid"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
    )
    return parser.parse_args()


def select_configuration(
    summary: pd.DataFrame,
    tolerance: float = 0.0025,
) -> pd.Series:
    eligible = summary[summary["eligible"]].copy()

    if eligible.empty:
        raise ValueError(
            "No configuration satisfies the predefined "
            "anti-collapse and clean-accuracy constraints."
        )

    best_score = eligible[
        "mean_validation_score"
    ].max()

    near_best = eligible[
        eligible["mean_validation_score"]
        >= best_score - tolerance
    ]

    return near_best.sort_values(
        ["lambda", "rho"],
        ascending=[True, True],
    ).iloc[0]


def main() -> None:
    args = parse_arguments()
    experiment = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )
    consistency_config = experiment["consistency"]

    lambda_candidates = [
        float(value)
        for value in consistency_config[
            "lambda_candidates"
        ]
    ]
    rho_candidates = [
        float(value)
        for value in consistency_config[
            "rho_candidates"
        ]
    ]
    development_seeds = {
        int(seed)
        for seed in consistency_config[
            "development_seeds"
        ]
    }

    expected_runs = len(development_seeds) * (
        1
        + len(lambda_candidates) * len(rho_candidates)
    )

    diagnostics_path = (
        args.root / "representation_diagnostics.csv"
    )
    if not diagnostics_path.exists():
        raise FileNotFoundError(diagnostics_path)

    diagnostics = pd.read_csv(diagnostics_path)
    rows = []

    config_paths = sorted(
        (args.root / "configs").glob("*.yaml")
    )
    allowed_run_counts = {
        expected_runs,
        expected_runs + len(development_seeds),
    }
    if len(config_paths) not in allowed_run_counts:
        raise ValueError(
            f"Expected one of {sorted(allowed_run_counts)} completed runs, "
            f"found {len(config_paths)}."
        )

    for config_path in config_paths:
        run = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        history_path = (
            args.root
            / "histories"
            / f"{run['run_name']}.csv"
        )
        if not history_path.exists():
            raise FileNotFoundError(history_path)

        history = pd.read_csv(history_path)
        best = history.sort_values(
            ["validation_score", "validation_loss", "epoch"],
            ascending=[False, True, True],
        ).iloc[0]

        rows.append(
            {
                "run_name": run["run_name"],
                "condition": run["condition"],
                "seed": int(run["seed"]),
                "lambda": float(
                    run["consistency_weight"]
                ),
                "rho": float(
                    run["variance_covariance_weight"]
                ),
                "best_epoch": int(best["epoch"]),
                "validation_score": float(
                    best["validation_score"]
                ),
                "clean_accuracy": float(
                    best["clean_accuracy"]
                ),
                "mean_shifted_accuracy": float(
                    best["mean_shifted_accuracy"]
                ),
                "worst_padding_shifted_accuracy": float(
                    best["worst_padding_shifted_accuracy"]
                ),
            }
        )

    runs = pd.DataFrame(rows)
    runs = runs.merge(
        diagnostics,
        on=[
            "run_name",
            "condition",
            "seed",
            "lambda",
            "rho",
        ],
        how="left",
        validate="one_to_one",
    )

    diagnostic_columns = [
        "effective_rank",
        "active_dimension_fraction",
        "effective_rank_retention",
        "active_dimension_fraction_retention",
    ]
    if runs[diagnostic_columns].isna().any().any():
        raise ValueError(
            "Representation diagnostics are incomplete."
        )

    exposure = runs[
        runs["condition"] == "exposure"
    ].set_index("seed")

    if set(exposure.index) != development_seeds:
        raise ValueError(
            "Exposure references do not match development seeds."
        )

    candidates = runs[
        runs["condition"] == "consistency"
    ].copy()

    candidates["clean_accuracy_drop_pp"] = [
        100
        * (
            float(exposure.loc[row.seed, "clean_accuracy"])
            - row.clean_accuracy
        )
        for row in candidates.itertuples(index=False)
    ]

    summary = (
        candidates.groupby(["lambda", "rho"])
        .agg(
            seeds=("seed", "nunique"),
            mean_validation_score=(
                "validation_score",
                "mean",
            ),
            std_validation_score=(
                "validation_score",
                "std",
            ),
            mean_clean_accuracy=(
                "clean_accuracy",
                "mean",
            ),
            mean_worst_padding_shifted_accuracy=(
                "worst_padding_shifted_accuracy",
                "mean",
            ),
            maximum_clean_accuracy_drop_pp=(
                "clean_accuracy_drop_pp",
                "max",
            ),
            minimum_effective_rank_retention=(
                "effective_rank_retention",
                "min",
            ),
            minimum_active_dimension_retention=(
                "active_dimension_fraction_retention",
                "min",
            ),
        )
        .reset_index()
    )

    base_configuration_count = (
        len(lambda_candidates) * len(rho_candidates)
    )
    allowed_configuration_counts = {
        base_configuration_count,
        base_configuration_count + 1,
    }
    if len(summary) not in allowed_configuration_counts:
        raise ValueError("The hyperparameter grid or boundary extension is incomplete.")

    if not (
        summary["seeds"] == len(development_seeds)
    ).all():
        raise ValueError(
            "Every configuration requires every development seed."
        )

    summary["eligible"] = (
        (
            summary["maximum_clean_accuracy_drop_pp"]
            <= consistency_config[
                "maximum_clean_accuracy_drop_pp"
            ]
        )
        & (
            summary["minimum_effective_rank_retention"]
            >= consistency_config[
                "minimum_effective_rank_retention"
            ]
        )
        & (
            summary[
                "minimum_active_dimension_retention"
            ]
            >= consistency_config[
                "minimum_active_dimension_fraction_retention"
            ]
        )
    )

    selected_row = select_configuration(summary)
    selected_lambda = float(selected_row["lambda"])
    selected_rho = float(selected_row["rho"])

    extensions = []
    if math.isclose(
        selected_lambda,
        max(lambda_candidates),
        abs_tol=1e-12,
    ):
        extensions.append(
            {
                "parameter": "lambda",
                "value": float(
                    consistency_config[
                        "boundary_extension"
                    ]["lambda_if_10_selected"]
                ),
            }
        )

    if math.isclose(
        selected_rho,
        max(rho_candidates),
        abs_tol=1e-12,
    ):
        extensions.append(
            {
                "parameter": "rho",
                "value": float(
                    consistency_config[
                        "boundary_extension"
                    ]["rho_if_0p2_selected"]
                ),
            }
        )

    def extension_was_evaluated(extension):
        if extension["parameter"] == "lambda":
            target_lambda = extension["value"]
            target_rho = selected_rho
        else:
            target_lambda = selected_lambda
            target_rho = extension["value"]

        return bool(
            (
                ((summary["lambda"] - target_lambda).abs() <= 1e-12)
                & ((summary["rho"] - target_rho).abs() <= 1e-12)
            ).any()
        )

    extensions = [
        extension
        for extension in extensions
        if not extension_was_evaluated(extension)
    ]

    selected = {
        "lambda": selected_lambda,
        "rho": selected_rho,
        "mean_validation_score": float(
            selected_row["mean_validation_score"]
        ),
        "mean_clean_accuracy": float(
            selected_row["mean_clean_accuracy"]
        ),
        "mean_worst_padding_shifted_accuracy": float(
            selected_row[
                "mean_worst_padding_shifted_accuracy"
            ]
        ),
        "maximum_clean_accuracy_drop_pp": float(
            selected_row[
                "maximum_clean_accuracy_drop_pp"
            ]
        ),
        "minimum_effective_rank_retention": float(
            selected_row[
                "minimum_effective_rank_retention"
            ]
        ),
        "minimum_active_dimension_retention": float(
            selected_row[
                "minimum_active_dimension_retention"
            ]
        ),
        "required_boundary_extensions": extensions,
    }

    candidates.to_csv(
        args.root / "consistency_grid_runs.csv",
        index=False,
    )
    summary.to_csv(
        args.root / "consistency_grid_summary.csv",
        index=False,
    )
    (args.root / "selected_consistency.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False),
        encoding="utf-8",
    )

    print(summary.round(6).to_string(index=False))
    print("\nSELECTED CONSISTENCY CONFIGURATION")
    print(yaml.safe_dump(selected, sort_keys=False))


if __name__ == "__main__":
    main()
