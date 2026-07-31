"""P1 multi-resolution preprocessing for the real RFID spectrum cache."""

from __future__ import annotations

import math

import numpy as np

from .spectrum import (
    COEFFICIENT_CHANNEL_COUNT,
    COEFFICIENT_COUNT,
    CONDITION_LIMIT,
    MINIMUM_DISTINCT_FREQUENCIES,
    RIDGE_ALPHA,
    SpectrumCoefficients,
    _derivative_basis,
    build_cubic_bspline_basis,
)


P1_SLOT_SEMANTICS = ("full_window", "early", "late")
_EXPECTED_TAIL = (3, 3, 50, 7)
_PHASOR_TOLERANCE = 1e-12


class PreprocessingContractError(ValueError):
    """A P1 preprocessing input or artifact violated its physical contract."""


def _validated_inputs(tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, observed = np.asarray(tensor), np.asarray(mask)
    if values.dtype != np.dtype("float32") or observed.dtype != np.dtype("bool"):
        raise PreprocessingContractError("P1 tensor and mask must be float32 and bool")
    if values.ndim != 5 or values.shape[1:] != _EXPECTED_TAIL:
        raise PreprocessingContractError("P1 tensor must have shape (N,3,3,50,7)")
    if observed.shape != values.shape[:-1] or values.shape[0] == 0:
        raise PreprocessingContractError("P1 mask must align with a nonempty tensor")
    if not np.isfinite(values).all():
        raise PreprocessingContractError("P1 tensor values must be finite")
    selected = values[observed]
    if selected.size:
        resultant = selected[:, 3:5]
        counts = np.expm1(selected[:, 5].astype(np.float64))
        phase_norm = np.hypot(selected[:, 0], selected[:, 1])
        if (
            np.any((resultant < 0.0) | (resultant > 1.0 + 1e-6))
            or np.any(counts <= 0.0)
            or np.any(np.abs(phase_norm - 1.0) > 2e-5)
        ):
            raise PreprocessingContractError("P1 observed physical channels are invalid")
    return values, observed


def _weighted_temporal_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return a vectorized weighted median over the three temporal cells."""

    temporal_values = np.moveaxis(values, 2, -1)
    temporal_weights = np.moveaxis(weights, 2, -1)
    order = np.argsort(temporal_values, axis=-1, kind="stable")
    sorted_values = np.take_along_axis(temporal_values, order, axis=-1)
    sorted_weights = np.take_along_axis(temporal_weights, order, axis=-1)
    total = sorted_weights.sum(axis=-1)
    threshold = 0.5 * total
    # Counts were stored as float32 log1p values.  Treat a cumulative mass
    # within float32 reconstruction tolerance of 50% as the weighted median
    # instead of non-deterministically jumping to the next temporal cell.
    tolerance = np.maximum(total * 2e-7, np.finfo(np.float64).eps)
    reached = np.cumsum(sorted_weights, axis=-1) + tolerance[..., None] >= threshold[..., None]
    chosen = np.argmax(reached, axis=-1)
    return np.take_along_axis(sorted_values, chosen[..., None], axis=-1)[..., 0]


def build_p1_multiresolution_tensor(
    tensor: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert `(early,middle,late)` into `(full-window,early,late)`.

    The full-window phase is a read-count and phase-coherence weighted
    circular aggregate. Scalar channels use a read-count weighted median.
    Masked values have exactly zero influence.
    """

    values, observed = _validated_inputs(tensor, mask)
    output = np.zeros_like(values)
    output_mask = np.zeros_like(observed)

    # Preserve endpoint dynamics while canonicalizing masked payloads to zero.
    output[:, :, 1] = np.where(observed[:, :, 0, :, None], values[:, :, 0], 0.0)
    output[:, :, 2] = np.where(observed[:, :, 2, :, None], values[:, :, 2], 0.0)
    output_mask[:, :, 1] = observed[:, :, 0]
    output_mask[:, :, 2] = observed[:, :, 2]

    counts = np.zeros(observed.shape, dtype=np.float64)
    counts[observed] = np.expm1(values[..., 5][observed].astype(np.float64))
    coherence_weight = counts * np.where(observed, values[..., 3].astype(np.float64), 0.0)
    real = np.sum(coherence_weight * np.where(observed, values[..., 1], 0.0), axis=2)
    imaginary = np.sum(coherence_weight * np.where(observed, values[..., 0], 0.0), axis=2)
    magnitude = np.hypot(real, imaginary)
    total_count = counts.sum(axis=2)
    full_valid = observed.any(axis=2) & (total_count > 0.0) & (magnitude > _PHASOR_TOLERANCE)

    full = output[:, :, 0]
    full[..., 0] = np.divide(
        imaginary,
        magnitude,
        out=np.zeros_like(imaginary),
        where=full_valid,
    )
    full[..., 1] = np.divide(
        real,
        magnitude,
        out=np.zeros_like(real),
        where=full_valid,
    )
    for channel in (2, 4, 6):
        full[..., channel] = _weighted_temporal_median(
            np.where(observed, values[..., channel], 0.0),
            counts,
        )
    full[..., 3] = np.divide(
        magnitude,
        total_count,
        out=np.zeros_like(magnitude),
        where=full_valid,
    )
    full[..., 5] = np.log1p(total_count)
    full[~full_valid] = 0.0
    output_mask[:, :, 0] = full_valid

    return output.astype(np.float32, copy=False), output_mask


def p2_quality_weights(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return fixed no-label frequency weights for P2 weighted splines."""

    values, observed = _validated_inputs(tensor, mask)
    counts = np.zeros(observed.shape, dtype=np.float64)
    gaps = np.zeros(observed.shape, dtype=np.float64)
    counts[observed] = np.expm1(values[..., 5][observed].astype(np.float64))
    gaps[observed] = np.expm1(values[..., 6][observed].astype(np.float64))
    if np.any(gaps[observed] < 0.0) or not np.isfinite(gaps[observed]).all():
        raise PreprocessingContractError("P2 observed read gaps are invalid")
    raw = np.zeros_like(counts)
    raw[observed] = (
        np.sqrt(counts[observed])
        * np.maximum(values[..., 3][observed].astype(np.float64), 0.05)
        / np.sqrt(1.0 + gaps[observed])
    )
    # Compute the ordinary median without NaN warnings for entirely masked
    # cells. With only 50 frequencies a stable sort is deterministic and
    # inexpensive relative to the subsequent spline solve.
    ordered = np.sort(np.where(observed, raw, np.inf), axis=-1, kind="stable")
    valid_count = observed.sum(axis=-1)
    low_index = np.maximum((valid_count - 1) // 2, 0)
    high_index = np.maximum(valid_count // 2, 0)
    low = np.take_along_axis(ordered, low_index[..., None], axis=-1)[..., 0]
    high = np.take_along_axis(ordered, high_index[..., None], axis=-1)[..., 0]
    median = np.where(valid_count > 0, 0.5 * (low + high), 1.0)
    if np.any(~np.isfinite(median[valid_count > 0])) or np.any(median[valid_count > 0] <= 0.0):
        raise PreprocessingContractError("P2 cell quality median is invalid")
    normalized = np.divide(
        raw,
        median[..., None],
        out=np.zeros_like(raw),
        where=observed,
    )
    normalized[observed] = np.clip(normalized[observed], 0.25, 4.0)
    return normalized


def extract_quality_weighted_spectrum(
    tensor: np.ndarray,
    mask: np.ndarray,
) -> SpectrumCoefficients:
    """Fit P2 cubic spectra using fixed read-quality observation weights."""

    values, observed = _validated_inputs(tensor, mask)
    weights = p2_quality_weights(values, observed)
    window_count = values.shape[0]
    basis = build_cubic_bspline_basis()
    derivative = _derivative_basis()
    cell_count = window_count * 3 * 3
    masks = observed.reshape(cell_count, 50)
    quality = weights.reshape(cell_count, 50)
    channels = values.reshape(cell_count, 50, COEFFICIENT_CHANNEL_COUNT).astype(np.float64, copy=False)
    coefficients = np.zeros((cell_count, COEFFICIENT_CHANNEL_COUNT, COEFFICIENT_COUNT), dtype=np.float64)
    availability = np.zeros(cell_count, dtype=np.float64)
    valid_count = masks.sum(axis=1).astype(np.int16)
    valid_fraction = valid_count.astype(np.float64) / 50.0
    conditions = np.full(cell_count, np.inf, dtype=np.float64)
    ranks = np.zeros(cell_count, dtype=np.int8)
    rmse = np.zeros((cell_count, COEFFICIENT_CHANNEL_COUNT), dtype=np.float64)
    derivative_energy = np.zeros((cell_count, COEFFICIENT_CHANNEL_COUNT), dtype=np.float64)
    ridge = RIDGE_ALPHA * np.eye(COEFFICIENT_COUNT, dtype=np.float64)

    for cell in np.flatnonzero(valid_count >= MINIMUM_DISTINCT_FREQUENCIES):
        valid = masks[cell]
        selected = basis[valid]
        rank = int(np.linalg.matrix_rank(selected))
        ranks[cell] = rank
        if rank != COEFFICIENT_COUNT:
            continue
        cell_weight = quality[cell, valid]
        normal = selected.T @ (cell_weight[:, None] * selected) + ridge
        condition = float(np.linalg.cond(normal))
        conditions[cell] = condition
        if not math.isfinite(condition) or condition > CONDITION_LIMIT:
            continue
        targets = channels[cell, valid]
        solved = np.linalg.solve(
            normal,
            selected.T @ (cell_weight[:, None] * targets),
        )
        coefficients[cell] = solved.T
        residual = selected @ solved - targets
        rmse[cell] = np.sqrt(
            np.sum(cell_weight[:, None] * residual * residual, axis=0)
            / cell_weight.sum()
        )
        slopes = derivative @ solved
        derivative_energy[cell] = np.mean(slopes * slopes, axis=0)
        availability[cell] = 1.0

    shape = (window_count, 3, 3)
    return SpectrumCoefficients(
        coefficients=coefficients.reshape(
            *shape,
            COEFFICIENT_CHANNEL_COUNT,
            COEFFICIENT_COUNT,
        ).astype(np.float32),
        availability=availability.reshape(shape),
        valid_count=valid_count.reshape(shape),
        valid_fraction=valid_fraction.reshape(shape),
        condition=conditions.reshape(shape),
        rank=ranks.reshape(shape),
        reconstruction_rmse=rmse.reshape(*shape, COEFFICIENT_CHANNEL_COUNT),
        derivative_energy=derivative_energy.reshape(*shape, COEFFICIENT_CHANNEL_COUNT),
    )

