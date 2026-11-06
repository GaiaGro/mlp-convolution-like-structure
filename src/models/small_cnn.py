import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

        if input_channels == output_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(
            self.main(inputs) + self.skip(inputs)
        )


class SmallCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (64, 128, 256, 384),
        blocks_per_stage: tuple[int, ...] = (1, 1, 2, 1),
    ) -> None:
        super().__init__()

        if len(channels) != len(blocks_per_stage):
            raise ValueError(
                "channels and blocks_per_stage must have equal length."
            )

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                channels[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        stages = []
        input_channels = channels[0]

        for stage_index, (output_channels, blocks) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            if blocks <= 0:
                raise ValueError(
                    "Every CNN stage requires at least one block."
                )

            for block_index in range(blocks):
                stride = (
                    2
                    if stage_index > 0 and block_index == 0
                    else 1
                )
                stages.append(
                    ResidualConvBlock(
                        input_channels=input_channels,
                        output_channels=output_channels,
                        stride=stride,
                    )
                )
                input_channels = output_channels

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_size = channels[-1]
        self.classifier = nn.Linear(
            self.feature_size,
            num_classes,
        )

    def forward_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.stem(images)
        features = self.stages(features)
        features = self.pool(features)
        return features.flatten(start_dim=1)

    def forward_with_features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        return self.classifier(features), features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(images)
        return logits
