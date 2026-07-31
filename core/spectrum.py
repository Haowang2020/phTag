"""Small B-spline helpers shared by preprocessing and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


COEFFICIENT_CHANNEL_COUNT = 7
COEFFICIENT_COUNT = 8
RIDGE_ALPHA = 1e-3
MINIMUM_DISTINCT_FREQUENCIES = 8
CONDITION_LIMIT = 1e8
_KNOTS = np.asarray(
    (0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SpectrumCoefficients:
    coefficients: np.ndarray
    availability: np.ndarray
    valid_count: np.ndarray
    valid_fraction: np.ndarray
    condition: np.ndarray
    rank: np.ndarray
    reconstruction_rmse: np.ndarray
    derivative_energy: np.ndarray


def build_cubic_bspline_basis() -> np.ndarray:
    points = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    identity = np.eye(COEFFICIENT_COUNT, dtype=np.float64)
    return np.column_stack(
        [
            BSpline(_KNOTS, identity[index], 3, extrapolate=False)(points)
            for index in range(COEFFICIENT_COUNT)
        ]
    )


def _derivative_basis() -> np.ndarray:
    points = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    identity = np.eye(COEFFICIENT_COUNT, dtype=np.float64)
    return np.column_stack(
        [
            BSpline(_KNOTS, identity[index], 3, extrapolate=False)
            .derivative()(points)
            for index in range(COEFFICIENT_COUNT)
        ]
    )
