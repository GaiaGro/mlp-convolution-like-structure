import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.translation import (
    evaluate_translation_robustness,
)
from src.training.engine import train_one_epoch


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 10),
        )

    def forward(self, images):
        return self.model(images)


class EngineTests(unittest.TestCase):
    def test_training_and_validation_are_finite(self):
        torch.manual_seed(1)

        images = torch.randn(16, 3, 8, 8)
        targets = torch.arange(16) % 10
        loader = DataLoader(
            TensorDataset(images, targets),
            batch_size=8,
        )

        model = TinyModel()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=False,
        )

        training = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            condition="consistency",
            consistency_weight=1.0,
            device=torch.device("cpu"),
            scaler=scaler,
            mixed_precision=False,
            translation_generator=(
                torch.Generator().manual_seed(1)
            ),
        )

        validation = evaluate_translation_robustness(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            shifts=((-1, 0), (1, 0)),
        )

        for value in training.values():
            self.assertTrue(
                torch.isfinite(torch.tensor(value)).item()
            )

        for value in validation.values():
            self.assertTrue(
                torch.isfinite(torch.tensor(value)).item()
            )


if __name__ == "__main__":
    unittest.main()
