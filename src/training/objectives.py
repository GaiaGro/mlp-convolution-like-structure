from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from src.data.transforms import (
    TRAIN_PADDING_MODES,
    TRAIN_SHIFTS,
    random_translate_batch,
)


CONDITIONS = {
    "baseline",
    "exposure",
    "consistency",
    "augmentation",
    "augmentation_consistency",
}

TWO_VIEW_CONDITIONS = {
    "exposure",
    "consistency",
    "augmentation_consistency",
}

CONSISTENCY_CONDITIONS = {
    "consistency",
    "augmentation_consistency",
}

AUGMENTED_CONDITIONS = {
    "augmentation",
    "augmentation_consistency",
}


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor
    classification_loss: torch.Tensor
    consistency_loss: torch.Tensor
    variance_covariance_loss: torch.Tensor
    variance_loss: torch.Tensor
    covariance_loss: torch.Tensor
    original_logits: torch.Tensor
    translated_logits: torch.Tensor | None


def jensen_shannon_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> torch.Tensor:
    if first_logits.shape != second_logits.shape:
        raise ValueError("Logit tensors must have equal shape.")

    first_log_probabilities = torch.log_softmax(
        first_logits,
        dim=1,
    )
    second_log_probabilities = torch.log_softmax(
        second_logits,
        dim=1,
    )

    first_probabilities = first_log_probabilities.exp()
    second_probabilities = second_log_probabilities.exp()

    mixture = 0.5 * (
        first_probabilities + second_probabilities
    )
    log_mixture = mixture.clamp_min(1e-12).log()

    first_divergence = (
        first_probabilities
        * (first_log_probabilities - log_mixture)
    ).sum(dim=1)

    second_divergence = (
        second_probabilities
        * (second_log_probabilities - log_mixture)
    ).sum(dim=1)

    return 0.5 * (
        first_divergence + second_divergence
    ).mean()


def variance_covariance_regularization(
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    variance_target: float = 1.0,
    epsilon: float = 1e-4,
    covariance_coefficient: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if first_features.shape != second_features.shape:
        raise ValueError("Feature tensors must have equal shape.")

    if first_features.ndim != 2:
        raise ValueError(
            "Expected feature tensors with shape "
            "(batch, dimensions)."
        )

    if first_features.shape[0] < 2:
        raise ValueError(
            "Variance-covariance regularization requires "
            "at least two samples."
        )

    if variance_target <= 0:
        raise ValueError("variance_target must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if covariance_coefficient < 0:
        raise ValueError(
            "covariance_coefficient must be non-negative."
        )

    first = first_features.float()
    second = second_features.float()

    def branch_terms(
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centered = features - features.mean(dim=0)
        standard_deviation = torch.sqrt(
            features.var(dim=0, unbiased=True) + epsilon
        )
        variance_term = torch.relu(
            variance_target - standard_deviation
        ).mean()

        covariance = (
            centered.T @ centered
        ) / (features.shape[0] - 1)

        dimensions = covariance.shape[0]
        off_diagonal_mask = ~torch.eye(
            dimensions,
            dtype=torch.bool,
            device=covariance.device,
        )
        covariance_term = (
            covariance[off_diagonal_mask]
            .square()
            .sum()
            / dimensions
        )
        return variance_term, covariance_term

    first_variance, first_covariance = branch_terms(first)
    second_variance, second_covariance = branch_terms(second)

    variance_loss = 0.5 * (
        first_variance + second_variance
    )
    covariance_loss = 0.5 * (
        first_covariance + second_covariance
    )
    combined = (
        variance_loss
        + covariance_coefficient * covariance_loss
    )

    return combined, variance_loss, covariance_loss


def compute_training_objective(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    condition: str,
    consistency_weight: float = 0.0,
    variance_covariance_weight: float = 0.0,
    variance_target: float = 1.0,
    epsilon: float = 1e-4,
    covariance_coefficient: float = 0.04,
    translation_generator: torch.Generator | None = None,
    paired_stochastic_masks: bool = True,
    translation_shifts: Sequence[tuple[int, int]] = TRAIN_SHIFTS,
    translation_padding_modes: Sequence[str] = TRAIN_PADDING_MODES,
) -> ObjectiveOutput:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    if consistency_weight < 0:
        raise ValueError(
            "consistency_weight must be non-negative."
        )
    if variance_covariance_weight < 0:
        raise ValueError(
            "variance_covariance_weight must be non-negative."
        )

    if (
        condition not in CONSISTENCY_CONDITIONS
        and (
            consistency_weight != 0
            or variance_covariance_weight != 0
        )
    ):
        raise ValueError(
            "Consistency and variance-covariance weights "
            "are only valid for consistency conditions."
        )

    paired_seed = None
    if (
        paired_stochastic_masks
        and condition in TWO_VIEW_CONDITIONS
    ):
        paired_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                generator=translation_generator,
            ).item()
        )

    def reset_paired_randomness() -> None:
        if paired_seed is not None:
            torch.manual_seed(paired_seed)

    reset_paired_randomness()
    require_features = variance_covariance_weight > 0

    if require_features:
        forward_with_features = getattr(
            model,
            "forward_with_features",
            None,
        )
        if forward_with_features is None:
            raise TypeError(
                "The model must implement forward_with_features "
                "when variance-covariance regularization is active."
            )
        original_logits, original_features = (
            forward_with_features(images)
        )
    else:
        original_logits = model(images)
        original_features = None

    original_loss = criterion(original_logits, targets)
    zero = original_loss.new_zeros(())

    if condition not in TWO_VIEW_CONDITIONS:
        return ObjectiveOutput(
            loss=original_loss,
            classification_loss=original_loss,
            consistency_loss=zero,
            variance_covariance_loss=zero,
            variance_loss=zero,
            covariance_loss=zero,
            original_logits=original_logits,
            translated_logits=None,
        )

    translated_images, _ = random_translate_batch(
        images,
        shifts=translation_shifts,
        padding_modes=translation_padding_modes,
        generator=translation_generator,
    )

    reset_paired_randomness()

    if require_features:
        translated_logits, translated_features = (
            model.forward_with_features(translated_images)
        )
    else:
        translated_logits = model(translated_images)
        translated_features = None

    translated_loss = criterion(
        translated_logits,
        targets,
    )

    classification_loss = 0.5 * (
        original_loss + translated_loss
    )

    if condition in CONSISTENCY_CONDITIONS:
        consistency_loss = jensen_shannon_divergence(
            original_logits,
            translated_logits,
        )
    else:
        consistency_loss = zero

    if require_features:
        (
            variance_covariance_loss,
            variance_loss,
            covariance_loss,
        ) = variance_covariance_regularization(
            original_features,
            translated_features,
            variance_target=variance_target,
            epsilon=epsilon,
            covariance_coefficient=covariance_coefficient,
        )
    else:
        variance_covariance_loss = zero
        variance_loss = zero
        covariance_loss = zero

    total_loss = (
        classification_loss
        + consistency_weight * consistency_loss
        + variance_covariance_weight
        * variance_covariance_loss
    )

    return ObjectiveOutput(
        loss=total_loss,
        classification_loss=classification_loss,
        consistency_loss=consistency_loss,
        variance_covariance_loss=variance_covariance_loss,
        variance_loss=variance_loss,
        covariance_loss=covariance_loss,
        original_logits=original_logits,
        translated_logits=translated_logits,
    )
