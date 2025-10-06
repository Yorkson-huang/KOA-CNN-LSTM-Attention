"""Objective function mirroring the MATLAB implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data_utils import load_wind_data
from .model import KOACNNLSTMAttention


def _round_positive(value: float, minimum: int = 1) -> int:
    rounded = int(round(value))
    return max(rounded, minimum)


class WindDataset(Dataset):
    """Tiny dataset representing day-level samples."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float64))
        self.targets = torch.from_numpy(targets.astype(np.float64))

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


@dataclass
class ObjectiveResult:
    loss_metric: float
    predictions: np.ndarray
    model: KOACNNLSTMAttention
    history: Sequence[float]
    metrics: dict


def objective_function(
    params: Sequence[float],
    *,
    data_path: str | None = None,
    device: torch.device | None = None,
) -> ObjectiveResult:
    """Train the network with specific hyper-parameters and report MAPE."""

    learning_rate = float(params[0])
    kernel_size = _round_positive(params[1])
    num_neurons = _round_positive(params[2])

    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure "same" padding behaves well, similar to MATLAB

    X_train, y_train, x_test, y_test = load_wind_data(data_path)

    dataset = WindDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=15, shuffle=True, drop_last=False)

    model = KOACNNLSTMAttention(kernel_size=kernel_size, num_neurons=num_neurons).double()
    device = device or torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )
    criterion = nn.MSELoss()

    history: list[float] = []
    model.train()
    for epoch in range(400):
        epoch_loss = 0.0
        total = 0
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = features.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size
        history.append(epoch_loss / max(total, 1))

    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(x_test.astype(np.float64)).unsqueeze(0).to(device)
        predictions = model(test_tensor).squeeze(0).cpu().numpy()

    error = predictions - y_test
    length = y_test.shape[0]
    sse = float(np.sum(np.square(error)))
    mae = float(np.sum(np.abs(error)) / length)
    mse = float(np.dot(error, error) / length)
    rmse = float(np.sqrt(mse))
    mean_target = float(np.mean(y_test))
    mean_denominator = mean_target if abs(mean_target) > 1e-8 else 1e-8
    mape = float(np.mean(np.abs(error / mean_denominator)))
    corr_matrix = np.corrcoef(y_test, predictions)
    r_value = float(corr_matrix[0, 1]) if corr_matrix.size >= 4 else float("nan")

    metrics = {
        "SSE": sse,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R": r_value,
    }

    return ObjectiveResult(
        loss_metric=mape,
        predictions=predictions,
        model=model,
        history=history,
        metrics=metrics,
    )

