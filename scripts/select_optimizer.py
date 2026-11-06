
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path("results/development/optimizer")


def main() -> None:
    rows = []

    for config_path in sorted((ROOT / "configs").glob("*.yaml")):
        run = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        history_path = (
            ROOT / "histories" / f"{run['run_name']}.csv"
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
                "model": run["model"],
                "learning_rate": run["learning_rate"],
                "weight_decay": run["weight_decay"],
                "seed": run["seed"],
                "best_epoch": int(best["epoch"]),
                "validation_score": best["validation_score"],
                "clean_accuracy": best["clean_accuracy"],
                "mean_shifted_accuracy": (
                    best["mean_shifted_accuracy"]
                ),
            }
        )

    runs = pd.DataFrame(rows)

    if len(runs) != 16:
        raise ValueError(
            f"Expected 16 completed runs, found {len(runs)}."
        )

    summary = (
        runs.groupby(
            ["model", "learning_rate", "weight_decay"]
        )
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
        raise ValueError("Every setting requires two seeds.")

    selected = {}

    for model, table in summary.groupby("model"):
        best = table.sort_values(
            [
                "mean_validation_score",
                "mean_clean_accuracy",
                "learning_rate",
                "weight_decay",
            ],
            ascending=[False, False, True, True],
        ).iloc[0]

        selected[model] = {
            "learning_rate": float(best["learning_rate"]),
            "weight_decay": float(best["weight_decay"]),
            "mean_validation_score": float(
                best["mean_validation_score"]
            ),
        }

    ROOT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(ROOT / "optimizer_runs.csv", index=False)
    summary.to_csv(
        ROOT / "optimizer_summary.csv",
        index=False,
    )
    (ROOT / "selected_optimizers.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False),
        encoding="utf-8",
    )

    print(summary.round(6).to_string(index=False))
    print("\nSELECTED OPTIMIZERS")
    print(yaml.safe_dump(selected, sort_keys=False))


if __name__ == "__main__":
    main()
