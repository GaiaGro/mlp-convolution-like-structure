from pathlib import Path

import yaml

from src.models import build_model, count_parameters


def main() -> None:
    config = yaml.safe_load(
        Path("configs/experiment.yaml").read_text(
            encoding="utf-8"
        )
    )

    mlp = build_model("mlp", config)
    cnn = build_model("cnn", config)

    mlp_parameters = count_parameters(mlp)
    cnn_parameters = count_parameters(cnn)
    relative_difference = abs(
        mlp_parameters - cnn_parameters
    ) / mlp_parameters

    print(f"MLP parameters: {mlp_parameters:,}")
    print(f"CNN parameters: {cnn_parameters:,}")
    print(
        "Relative parameter difference: "
        f"{100 * relative_difference:.3f}%"
    )

    if relative_difference > 0.10:
        raise AssertionError(
            "CNN and MLP differ by more than 10%."
        )


if __name__ == "__main__":
    main()
