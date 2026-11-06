import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        width: int,
        expansion_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        expanded_width = width * expansion_ratio

        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, expanded_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 3 * 32 * 32,
        num_classes: int = 10,
        hidden_width: int = 512,
        residual_blocks: int = 3,
        expansion_ratio: int = 2,
        dropout: float = 0.1,
        pixel_permutation_seed: int | None = None,
    ) -> None:
        super().__init__()

        if (
            pixel_permutation_seed is not None
            and pixel_permutation_seed < 0
        ):
            raise ValueError(
                "pixel_permutation_seed must be non-negative."
            )

        if (
            pixel_permutation_seed is not None
            and input_size != 3 * 32 * 32
        ):
            raise ValueError(
                "Pixel permutation is defined for CIFAR-10 inputs."
            )

        self.input_size = input_size
        self.hidden_width = hidden_width
        self.pixel_permutation_seed = pixel_permutation_seed

        if pixel_permutation_seed is None:
            permutation = torch.arange(32 * 32)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(pixel_permutation_seed)
            permutation = torch.randperm(
                32 * 32,
                generator=generator,
            )

        self.register_buffer(
            "pixel_permutation",
            permutation,
            persistent=False,
        )

        self.input_projection = nn.Linear(
            input_size,
            hidden_width,
        )
        self.input_norm = nn.LayerNorm(hidden_width)
        self.input_activation = nn.GELU()

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    width=hidden_width,
                    expansion_ratio=expansion_ratio,
                    dropout=dropout,
                )
                for _ in range(residual_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_width)
        self.classifier = nn.Linear(hidden_width, num_classes)

    def forward_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.pixel_permutation_seed is not None:
            if images.ndim != 4 or images.shape[1:] != (
                3,
                32,
                32,
            ):
                raise ValueError(
                    "Pixel-permuted MLP expects images with shape "
                    "(batch, 3, 32, 32)."
                )

            images = images.flatten(start_dim=2).index_select(
                2,
                self.pixel_permutation,
            ).reshape_as(images)

        flattened = images.flatten(start_dim=1)

        if flattened.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input values, "
                f"found {flattened.shape[1]}."
            )

        features = self.input_projection(flattened)
        features = self.input_norm(features)
        features = self.input_activation(features)
        features = self.blocks(features)
        return self.output_norm(features)

    def forward_with_features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        return self.classifier(features), features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(images)
        return logits
