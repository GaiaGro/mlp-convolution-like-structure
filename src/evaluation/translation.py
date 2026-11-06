from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.transforms import translate_batch


VALIDATION_SHIFTS = (
    (-1, 0),
    (1, 0),
    (-2, 0),
    (2, 0),
    (0, -1),
    (0, 1),
    (0, -2),
    (0, 2),
)

PRIMARY_PADDING_MODES = (
    "reflect",
    "replicate",
    "circular",
)


def build_axial_shifts(
    distances: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    distances = tuple(int(value) for value in distances)

    if not distances:
        raise ValueError("At least one distance is required.")
    if any(value <= 0 for value in distances):
        raise ValueError("Translation distances must be positive.")
    if len(set(distances)) != len(distances):
        raise ValueError("Translation distances must be unique.")

    return tuple(
        shift
        for distance in distances
        for shift in (
            (-distance, 0),
            (distance, 0),
            (0, -distance),
            (0, distance),
        )
    )


@torch.inference_mode()
def evaluate_translation_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    shifts=VALIDATION_SHIFTS,
    padding_modes: Sequence[str] = PRIMARY_PADDING_MODES,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    if not shifts:
        raise ValueError("At least one shift is required.")
    if not padding_modes:
        raise ValueError("At least one padding mode is required.")

    samples = 0
    clean_correct = 0
    clean_loss = 0.0

    combinations = [
        (padding_mode, shift)
        for padding_mode in padding_modes
        for shift in shifts
    ]

    shifted_correct = {
        combination: 0 for combination in combinations
    }
    consistent_predictions = {
        combination: 0 for combination in combinations
    }

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        clean_logits = model(images)
        clean_predictions = clean_logits.argmax(dim=1)

        batch_size = targets.shape[0]
        samples += batch_size
        clean_correct += (
            clean_predictions == targets
        ).sum().item()
        clean_loss += (
            criterion(clean_logits, targets).item()
            * batch_size
        )

        for padding_mode, shift in combinations:
            dx, dy = shift
            translated_images = translate_batch(
                images,
                dx=dx,
                dy=dy,
                padding_mode=padding_mode,
            )
            translated_logits = model(translated_images)
            translated_predictions = translated_logits.argmax(
                dim=1
            )

            key = (padding_mode, shift)
            shifted_correct[key] += (
                translated_predictions == targets
            ).sum().item()
            consistent_predictions[key] += (
                translated_predictions == clean_predictions
            ).sum().item()

    if samples == 0:
        raise ValueError("Validation loader is empty.")

    clean_accuracy = clean_correct / samples

    mode_shifted_accuracies = {}
    mode_consistencies = {}

    for padding_mode in padding_modes:
        shifted_values = [
            shifted_correct[(padding_mode, shift)] / samples
            for shift in shifts
        ]
        consistency_values = [
            consistent_predictions[(padding_mode, shift)]
            / samples
            for shift in shifts
        ]

        mode_shifted_accuracies[padding_mode] = (
            sum(shifted_values) / len(shifted_values)
        )
        mode_consistencies[padding_mode] = (
            sum(consistency_values)
            / len(consistency_values)
        )

    mean_shifted_accuracy = (
        sum(mode_shifted_accuracies.values())
        / len(mode_shifted_accuracies)
    )
    worst_padding_shifted_accuracy = min(
        mode_shifted_accuracies.values()
    )

    mean_consistency = (
        sum(mode_consistencies.values())
        / len(mode_consistencies)
    )
    worst_padding_consistency = min(
        mode_consistencies.values()
    )

    validation_score = 0.5 * (
        clean_accuracy + worst_padding_shifted_accuracy
    )

    metrics = {
        "validation_loss": clean_loss / samples,
        "clean_accuracy": clean_accuracy,
        "mean_shifted_accuracy": mean_shifted_accuracy,
        "worst_padding_shifted_accuracy": (
            worst_padding_shifted_accuracy
        ),
        "mean_prediction_consistency": mean_consistency,
        "worst_padding_prediction_consistency": (
            worst_padding_consistency
        ),
        "validation_score": validation_score,
    }

    for padding_mode in padding_modes:
        metrics[
            f"{padding_mode}_mean_shifted_accuracy"
        ] = mode_shifted_accuracies[padding_mode]
        metrics[
            f"{padding_mode}_mean_prediction_consistency"
        ] = mode_consistencies[padding_mode]

    return metrics
