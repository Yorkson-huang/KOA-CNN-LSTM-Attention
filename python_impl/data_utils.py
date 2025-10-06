"""Data loading utilities for the KOA-CNN-LSTM-Attention Python port."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
PRIMARY_DATA_FILE = PACKAGE_DIR / "Data.xlsx"
LEGACY_ENGLISH_NAME = "FeatureSequenceAndActual.xlsx"
FALLBACK_FILES = (
    PRIMARY_DATA_FILE,
    Path("Data.xlsx"),
    PACKAGE_DIR / LEGACY_ENGLISH_NAME,
    Path(LEGACY_ENGLISH_NAME),
)


def _resolve_data_path(data_path: str | None) -> Path:
    """Resolve the Excel data path, preferring the packaged Data.xlsx file."""
    if data_path:
        provided = Path(data_path)
        if provided.exists():
            return provided
        raise FileNotFoundError(f"Provided data file not found: {provided}")

    for candidate in FALLBACK_FILES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset not found. Place Data.xlsx inside python_impl/ or specify --data-path."
    )


def _clean_numeric_block(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove textual headers/labels and keep only numeric values."""
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(axis=0, how="all")
    numeric_frame = numeric_frame.dropna(axis=1, how="all")

    if numeric_frame.isna().any().any():
        raise ValueError(
            "Dataset contains non-numeric entries beyond header/label rows. "
            "Please ensure all feature/target values are numeric."
        )
    return numeric_frame


def load_wind_data(
    data_path: str | None = None,
    *,
    num_days: int = 75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and reshape wind-speed data to match the MATLAB logic.

    Returns
    -------
    X_train : np.ndarray
        Training inputs shaped (73, 18, 24).
    y_train : np.ndarray
        Training targets shaped (73, 24).
    x_test : np.ndarray
        Test input shaped (18, 24).
    y_test : np.ndarray
        Test target shaped (24,).
    """
    excel_path = _resolve_data_path(data_path)
    frame = pd.read_excel(excel_path, header=None)
    numeric_frame = _clean_numeric_block(frame)
    data = numeric_frame.to_numpy(dtype=np.float32)

    if data.shape[0] < 19:
        raise ValueError(
            "Expected at least 19 rows in the Excel sheet (18 features + 1 target)."
        )

    features = data[:18, :]
    wind = data[18, :]

    expected_cols = 24 * num_days
    if features.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns for {num_days} days, got {features.shape[1]}."
        )

    features_reshaped = np.reshape(features, (18, 24, 1, num_days), order="F")
    wind_reshaped = np.reshape(wind, (24, 1, 1, num_days), order="F")

    feature_days = [features_reshaped[:, :, 0, i].astype(np.float32) for i in range(num_days)]
    wind_days = [wind_reshaped[:, 0, 0, i].astype(np.float32) for i in range(num_days)]

    X_train = np.stack(feature_days[:73], axis=0)
    y_train = np.stack(wind_days[1:74], axis=0)

    x_test = feature_days[73]
    y_test = wind_days[74]

    return X_train, y_train, x_test, y_test

