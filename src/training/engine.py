import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.transforms import (
    TRAIN_PADDING_MODES,
    TRAIN_SHIFTS,
)

from src.training.objectives import (
    compute_training_objective,
)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
):
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")

    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError(
            "warmup_epochs must be between 0 and total_epochs."
        )

    def learning_rate_multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (
            epoch - warmup_epochs
        ) / max(1, total_epochs - warmup_epochs - 1)

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    condition: str,
    consistency_weight: float,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    translation_generator: torch.Generator,
    variance_covariance_weight: float = 0.0,
    variance_target: float = 1.0,
    epsilon: float = 1e-4,
    covariance_coefficient: float = 0.04,
    paired_stochastic_masks: bool = True,
    translation_shifts: Sequence[tuple[int, int]] = TRAIN_SHIFTS,
    translation_padding_modes: Sequence[str] = TRAIN_PADDING_MODES,
) -> dict[str, float]:
    model.train()

    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "consistency_loss": 0.0,
        "variance_covariance_loss": 0.0,
        "variance_loss": 0.0,
        "covariance_loss": 0.0,
        "original_correct": 0,
        "translated_correct": 0,
        "translated_samples": 0,
        "samples": 0,
    }

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=mixed_precision,
        ):
            output = compute_training_objective(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                condition=condition,
                consistency_weight=consistency_weight,
                variance_covariance_weight=(
                    variance_covariance_weight
                ),
                variance_target=variance_target,
                epsilon=epsilon,
                covariance_coefficient=(
                    covariance_coefficient
                ),
                translation_generator=translation_generator,
                paired_stochastic_masks=paired_stochastic_masks,
                translation_shifts=translation_shifts,
                translation_padding_modes=translation_padding_modes,
            )

        if not torch.isfinite(output.loss):
            raise FloatingPointError(
                "Non-finite training loss detected."
            )

        scaler.scale(output.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        totals["samples"] += batch_size
        totals["loss"] += output.loss.item() * batch_size
        totals["classification_loss"] += (
            output.classification_loss.item() * batch_size
        )
        totals["consistency_loss"] += (
            output.consistency_loss.item() * batch_size
        )
        totals["variance_covariance_loss"] += (
            output.variance_covariance_loss.item()
            * batch_size
        )
        totals["variance_loss"] += (
            output.variance_loss.item() * batch_size
        )
        totals["covariance_loss"] += (
            output.covariance_loss.item() * batch_size
        )
        totals["original_correct"] += (
            output.original_logits.argmax(dim=1) == targets
        ).sum().item()

        if output.translated_logits is not None:
            totals["translated_samples"] += batch_size
            totals["translated_correct"] += (
                output.translated_logits.argmax(dim=1)
                == targets
            ).sum().item()

    samples = totals["samples"]
    translated_samples = totals["translated_samples"]

    return {
        "train_loss": totals["loss"] / samples,
        "train_classification_loss": (
            totals["classification_loss"] / samples
        ),
        "train_consistency_loss": (
            totals["consistency_loss"] / samples
        ),
        "train_variance_covariance_loss": (
            totals["variance_covariance_loss"] / samples
        ),
        "train_variance_loss": (
            totals["variance_loss"] / samples
        ),
        "train_covariance_loss": (
            totals["covariance_loss"] / samples
        ),
        "train_original_accuracy": (
            totals["original_correct"] / samples
        ),
        "train_translated_accuracy": (
            totals["translated_correct"] / translated_samples
            if translated_samples
            else float("nan")
        ),
    }
