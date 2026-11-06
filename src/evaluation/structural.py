import torch


def validate_first_layer_weights(
    weights: torch.Tensor,
) -> None:
    if weights.ndim != 4:
        raise ValueError(
            "Expected weights with shape "
            "(neurons, channels, height, width)."
        )

    if weights.shape[1] != 3:
        raise ValueError("Expected three RGB channels.")

    if not weights.is_floating_point():
        raise TypeError("Weights must be floating point.")

    total_energy = weights.square().sum(dim=(1, 2, 3))
    if (total_energy <= 0).any():
        raise ValueError("Every neuron must have positive energy.")


def spatial_energy(
    weights: torch.Tensor,
) -> torch.Tensor:
    validate_first_layer_weights(weights)
    return weights.square().sum(dim=1)


def maximum_window_energy_fraction(
    weights: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_first_layer_weights(weights)

    height, width = weights.shape[-2:]

    if (
        not isinstance(window_size, int)
        or window_size <= 0
        or window_size > min(height, width)
    ):
        raise ValueError("Invalid window size.")

    energy = spatial_energy(weights)
    windows = (
        energy.unfold(1, window_size, 1)
        .unfold(2, window_size, 1)
        .sum(dim=(-1, -2))
    )

    flattened = windows.flatten(start_dim=1)
    maximum_energy, indices = flattened.max(dim=1)

    output_width = width - window_size + 1
    rows = torch.div(
        indices,
        output_width,
        rounding_mode="floor",
    )
    columns = indices.remainder(output_width)
    locations = torch.stack((rows, columns), dim=1)

    total_energy = energy.sum(dim=(1, 2))
    fractions = maximum_energy / total_energy

    return fractions, locations


def extract_maximum_energy_patches(
    weights: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fractions, locations = maximum_window_energy_fraction(
        weights,
        window_size,
    )

    patches = []

    for index, location in enumerate(locations):
        row = int(location[0].item())
        column = int(location[1].item())
        patches.append(
            weights[
                index,
                :,
                row : row + window_size,
                column : column + window_size,
            ]
        )

    return torch.stack(patches), fractions, locations


def shuffle_spatial_weights(
    weights: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    validate_first_layer_weights(weights)

    neurons, channels, height, width = weights.shape
    pixels = height * width
    flattened = weights.reshape(neurons, channels, pixels)
    shuffled = torch.empty_like(flattened)

    for neuron in range(neurons):
        permutation = torch.randperm(
            pixels,
            generator=generator,
            device="cpu",
        ).to(weights.device)

        shuffled[neuron] = flattened[neuron].index_select(
            1,
            permutation,
        )

    return shuffled.reshape(neurons, channels, height, width)


def weights_in_original_image_coordinates(
    weights: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    validate_first_layer_weights(weights)

    neurons, channels, height, width = weights.shape
    pixels = height * width

    if permutation.ndim != 1 or permutation.numel() != pixels:
        raise ValueError("Permutation has an invalid shape.")

    permutation_cpu = permutation.detach().to(
        device="cpu",
        dtype=torch.long,
    )

    expected = torch.arange(pixels)
    if not torch.equal(
        permutation_cpu.sort().values,
        expected,
    ):
        raise ValueError(
            "Permutation must contain every spatial index once."
        )

    flattened = weights.reshape(neurons, channels, pixels)
    remapped = torch.empty_like(flattened)
    remapped.index_copy_(
        2,
        permutation_cpu.to(weights.device),
        flattened,
    )

    return remapped.reshape(neurons, channels, height, width)

def fit_sign_invariant_spherical_kmeans(
    samples: torch.Tensor,
    template_count: int,
    initializations: int,
    maximum_iterations: int,
    tolerance: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError("Samples must be a matrix.")

    sample_count = samples.shape[0]

    if not 1 <= template_count <= sample_count:
        raise ValueError("Invalid template count.")
    if initializations <= 0:
        raise ValueError("Initializations must be positive.")
    if maximum_iterations <= 0:
        raise ValueError("Maximum iterations must be positive.")
    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative.")

    norms = samples.norm(dim=1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("Every sample must have positive norm.")

    normalized = samples / norms
    best_templates = None
    best_objective = float("-inf")

    for _ in range(initializations):
        initial_indices = torch.randperm(
            sample_count,
            generator=generator,
            device=samples.device,
        )[:template_count]

        templates = normalized.index_select(
            0,
            initial_indices,
        ).clone()

        for _ in range(maximum_iterations):
            similarities = normalized @ templates.T
            assignments = similarities.abs().argmax(dim=1)

            selected = similarities.gather(
                1,
                assignments.unsqueeze(1),
            ).squeeze(1)

            signs = torch.where(
                selected < 0,
                -torch.ones_like(selected),
                torch.ones_like(selected),
            )

            updated = []

            for template_index in range(template_count):
                mask = assignments == template_index

                if mask.any():
                    candidate = (
                        normalized[mask]
                        * signs[mask].unsqueeze(1)
                    ).mean(dim=0)
                else:
                    replacement = torch.randint(
                        sample_count,
                        (1,),
                        generator=generator,
                        device=samples.device,
                    )
                    candidate = normalized[
                        replacement.item()
                    ]

                candidate_norm = candidate.norm()
                if candidate_norm <= 0:
                    raise FloatingPointError(
                        "A template collapsed to zero."
                    )

                updated.append(candidate / candidate_norm)

            updated_templates = torch.stack(updated)

            change = (
                1.0
                - (
                    templates * updated_templates
                ).sum(dim=1).abs()
            ).max()

            templates = updated_templates

            if change.item() <= tolerance:
                break

        objective = (
            (normalized @ templates.T)
            .abs()
            .max(dim=1)
            .values
            .square()
            .mean()
            .item()
        )

        if objective > best_objective:
            best_objective = objective
            best_templates = templates.clone()

    if best_templates is None:
        raise RuntimeError("Template fitting failed.")

    return best_templates

def translated_template_scores(
    weights: torch.Tensor,
    templates: torch.Tensor,
    patch_size: int,
) -> dict[str, torch.Tensor]:
    validate_first_layer_weights(weights)

    if templates.ndim != 2:
        raise ValueError("Templates must be a matrix.")

    channels = weights.shape[1]
    expected_dimension = channels * patch_size * patch_size

    if templates.shape[1] != expected_dimension:
        raise ValueError("Template dimension is incompatible.")

    template_norms = templates.norm(dim=1)
    if not torch.allclose(
        template_norms,
        torch.ones_like(template_norms),
        atol=1e-5,
    ):
        raise ValueError("Templates must have unit norm.")

    height, width = weights.shape[-2:]
    if patch_size <= 0 or patch_size > min(height, width):
        raise ValueError("Invalid patch size.")

    patches = (
        weights.unfold(2, patch_size, 1)
        .unfold(3, patch_size, 1)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(
            weights.shape[0],
            -1,
            expected_dimension,
        )
    )

    similarities = torch.einsum(
        "npd,kd->npk",
        patches,
        templates,
    )
    squared_similarities = similarities.square()
    flattened = squared_similarities.flatten(start_dim=1)

    best_reconstruction, best_indices = flattened.max(
        dim=1
    )

    template_count = templates.shape[0]
    position_indices = torch.div(
        best_indices,
        template_count,
        rounding_mode="floor",
    )

    patch_energy = patches.square().sum(dim=2)
    selected_patch_energy = patch_energy.gather(
        1,
        position_indices.unsqueeze(1),
    ).squeeze(1)

    total_energy = weights.square().sum(dim=(1, 2, 3))

    conditional_score = (
        best_reconstruction
        / selected_patch_energy.clamp_min(1e-12)
    ).clamp(0.0, 1.0)

    total_reconstruction_fraction = (
        best_reconstruction
        / total_energy.clamp_min(1e-12)
    ).clamp(0.0, 1.0)

    return {
        "conditional_shape_score": conditional_score,
        "total_energy_reconstruction_fraction": (
            total_reconstruction_fraction
        ),
        "selected_patch_energy_fraction": (
            selected_patch_energy / total_energy
        ).clamp(0.0, 1.0),
    }


def cross_validated_template_reuse(
    weights: torch.Tensor,
    patch_size: int,
    template_count: int,
    folds: int,
    repetitions: int,
    initializations: int,
    maximum_iterations: int,
    tolerance: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    validate_first_layer_weights(weights)

    neuron_count = weights.shape[0]

    if not 2 <= folds <= neuron_count:
        raise ValueError("Invalid number of folds.")
    if repetitions <= 0:
        raise ValueError("Repetitions must be positive.")

    largest_test_fold = (
        neuron_count + folds - 1
    ) // folds
    minimum_training_size = (
        neuron_count - largest_test_fold
    )

    if template_count > minimum_training_size:
        raise ValueError(
            "Template count exceeds a training fold."
        )

    maximum_patches, locality, _ = (
        extract_maximum_energy_patches(
            weights,
            patch_size,
        )
    )
    patch_samples = maximum_patches.flatten(start_dim=1)

    generator = torch.Generator(device=weights.device)
    generator.manual_seed(seed)

    accumulated_conditional = torch.zeros(
        neuron_count,
        device=weights.device,
        dtype=weights.dtype,
    )
    accumulated_total = torch.zeros_like(
        accumulated_conditional
    )
    accumulated_selected_energy = torch.zeros_like(
        accumulated_conditional
    )

    for _ in range(repetitions):
        ordering = torch.randperm(
            neuron_count,
            generator=generator,
            device=weights.device,
        )

        fold_ids = torch.empty(
            neuron_count,
            dtype=torch.long,
            device=weights.device,
        )
        fold_ids[ordering] = (
            torch.arange(
                neuron_count,
                device=weights.device,
            )
            % folds
        )

        repetition_conditional = torch.empty_like(
            accumulated_conditional
        )
        repetition_total = torch.empty_like(
            accumulated_total
        )
        repetition_selected_energy = torch.empty_like(
            accumulated_selected_energy
        )

        for fold in range(folds):
            test_mask = fold_ids == fold
            train_mask = ~test_mask

            templates = (
                fit_sign_invariant_spherical_kmeans(
                    samples=patch_samples[train_mask],
                    template_count=template_count,
                    initializations=initializations,
                    maximum_iterations=(
                        maximum_iterations
                    ),
                    tolerance=tolerance,
                    generator=generator,
                )
            )

            scores = translated_template_scores(
                weights=weights[test_mask],
                templates=templates,
                patch_size=patch_size,
            )

            repetition_conditional[test_mask] = scores[
                "conditional_shape_score"
            ]
            repetition_total[test_mask] = scores[
                "total_energy_reconstruction_fraction"
            ]
            repetition_selected_energy[test_mask] = scores[
                "selected_patch_energy_fraction"
            ]

        accumulated_conditional += (
            repetition_conditional
        )
        accumulated_total += repetition_total
        accumulated_selected_energy += (
            repetition_selected_energy
        )

    scale = float(repetitions)

    return {
        "conditional_shape_score": (
            accumulated_conditional / scale
        ),
        "total_energy_reconstruction_fraction": (
            accumulated_total / scale
        ),
        "selected_patch_energy_fraction": (
            accumulated_selected_energy / scale
        ),
        "maximum_patch_energy_fraction": locality,
    }
