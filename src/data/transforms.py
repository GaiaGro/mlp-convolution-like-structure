from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TRAIN_SHIFTS = (
    (-2, 0),
    (-1, 0),
    (1, 0),
    (2, 0),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
)

TRAIN_PADDING_MODES = (
    "reflect",
    "replicate",
    "circular",
)

SUPPORTED_PADDING_MODES = (
    *TRAIN_PADDING_MODES,
    "zero",
)


def build_cifar10_transforms(use_augmentation: bool):
    evaluation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    if not use_augmentation:
        return evaluation, evaluation

    training = transforms.Compose(
        [
            transforms.RandomCrop(
                32,
                padding=4,
                padding_mode="reflect",
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return training, evaluation


def _zero_pad_normalized_cifar10(
    images: torch.Tensor,
    padding: int,
) -> torch.Tensor:
    if images.shape[1] != 3:
        raise ValueError(
            "Zero padding requires three-channel CIFAR-10 tensors."
        )

    fill = images.new_tensor(
        [
            -mean / std
            for mean, std in zip(CIFAR10_MEAN, CIFAR10_STD)
        ]
    ).view(1, 3, 1, 1)

    batch, channels, height, width = images.shape
    padded = fill.expand(
        batch,
        channels,
        height + 2 * padding,
        width + 2 * padding,
    ).clone()

    padded[
        :,
        :,
        padding : padding + height,
        padding : padding + width,
    ] = images

    return padded


def translate_batch(
    images: torch.Tensor,
    dx: int,
    dy: int,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """Translate normalized CIFAR-10 tensors without interpolation."""

    if images.ndim != 4:
        raise ValueError(
            "Expected (batch, channels, height, width)."
        )

    if padding_mode not in SUPPORTED_PADDING_MODES:
        raise ValueError(
            f"Unsupported padding mode: {padding_mode}."
        )

    padding = max(abs(dx), abs(dy))
    if padding == 0:
        return images.clone()

    if padding >= min(images.shape[-2:]):
        raise ValueError("Translation exceeds image dimensions.")

    if padding_mode == "zero":
        padded = _zero_pad_normalized_cifar10(
            images,
            padding,
        )
    else:
        padded = F.pad(
            images,
            (padding, padding, padding, padding),
            mode=padding_mode,
        )

    height, width = images.shape[-2:]
    start_x = padding - dx
    start_y = padding - dy

    return padded[
        :,
        :,
        start_y : start_y + height,
        start_x : start_x + width,
    ]


def random_translate_batch(
    images: torch.Tensor,
    shifts: Sequence[tuple[int, int]] = TRAIN_SHIFTS,
    padding_modes: Sequence[str] = TRAIN_PADDING_MODES,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomize both translation and boundary condition per sample."""

    if not shifts:
        raise ValueError("At least one shift is required.")

    if not padding_modes:
        raise ValueError("At least one padding mode is required.")

    invalid_modes = set(padding_modes) - set(
        SUPPORTED_PADDING_MODES
    )
    if invalid_modes:
        raise ValueError(
            f"Unsupported padding modes: {sorted(invalid_modes)}."
        )

    shift_choices = torch.randint(
        0,
        len(shifts),
        (images.shape[0],),
        generator=generator,
        device="cpu",
    )
    mode_choices = torch.randint(
        0,
        len(padding_modes),
        (images.shape[0],),
        generator=generator,
        device="cpu",
    )

    translated = torch.empty_like(images)
    device_shift_choices = shift_choices.to(images.device)
    device_mode_choices = mode_choices.to(images.device)

    for shift_index, (dx, dy) in enumerate(shifts):
        for mode_index, padding_mode in enumerate(
            padding_modes
        ):
            mask = (
                (device_shift_choices == shift_index)
                & (device_mode_choices == mode_index)
            )
            if mask.any():
                translated[mask] = translate_batch(
                    images[mask],
                    dx=dx,
                    dy=dy,
                    padding_mode=padding_mode,
                )

    return translated, shift_choices
