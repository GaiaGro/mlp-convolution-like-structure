from torch import nn

from src.models.residual_mlp import ResidualMLP
from src.models.small_cnn import SmallCNN


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_model(
    model_name: str,
    config: dict,
    pixel_permutation_seed: int | None = None,
) -> nn.Module:
    if model_name == "mlp":
        model_config = config["mlp"]
        return ResidualMLP(
            hidden_width=model_config["hidden_width"],
            residual_blocks=model_config["residual_blocks"],
            expansion_ratio=model_config["expansion_ratio"],
            dropout=model_config["dropout"],
            pixel_permutation_seed=pixel_permutation_seed,
        )

    if model_name == "cnn":
        if pixel_permutation_seed is not None:
            raise ValueError(
                "Pixel permutation is only defined for the MLP."
            )

        model_config = config["cnn"]
        return SmallCNN(
            channels=tuple(model_config["channels"]),
            blocks_per_stage=tuple(
                model_config["blocks_per_stage"]
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")
