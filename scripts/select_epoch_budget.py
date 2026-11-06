import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the MLP training budget."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/development/epoch_ablation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    root = args.root
    rows = []

    for config_path in sorted((root / "configs").glob("*.yaml")):
        run = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        history_path = (
            root / "histories" / f"{run['run_name']}.csv"
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
                "epochs": int(run["epochs"]),
                "seed": int(run["seed"]),
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
            }
        )

    runs = pd.DataFrame(rows)

    if len(runs) != 6:
        raise ValueError(
            f"Expected 6 completed runs, found {len(runs)}."
        )

    summary = (
        runs.groupby("epochs")
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
            mean_shifted_accuracy=(
                "mean_shifted_accuracy",
                "mean",
            ),
        )
        .reset_index()
    )

    if not (summary["seeds"] == 2).all():
        raise ValueError("Every budget requires two seeds.")

    best_score = float(
        summary["mean_validation_score"].max()
    )
    tolerance = 0.0025
    eligible = summary[
        summary["mean_validation_score"]
        >= best_score - tolerance
    ].sort_values("epochs")

    selected_row = eligible.iloc[0]
    selected = {
        "epochs": int(selected_row["epochs"]),
        "mean_validation_score": float(
            selected_row["mean_validation_score"]
        ),
        "best_observed_mean_validation_score": best_score,
        "shorter_budget_tolerance": tolerance,
    }

    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(root / "epoch_ablation_runs.csv", index=False)
    summary.to_csv(
        root / "epoch_ablation_summary.csv",
        index=False,
    )
    (root / "selected_epoch_budget.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False),
        encoding="utf-8",
    )

    print(summary.round(6).to_string(index=False))
    print("\nSELECTED EPOCH BUDGET")
    print(yaml.safe_dump(selected, sort_keys=False))


if __name__ == "__main__":
    main()
