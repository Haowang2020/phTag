"""Session-level physical features for the real-RFID Q3-1C0 ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import numpy as np

from .spectrum import (
    COEFFICIENT_CHANNEL_COUNT,
    COEFFICIENT_COUNT,
    SpectrumCoefficients,
    build_cubic_bspline_basis,
)

BRANCH_NAMES = ("d4", "sensing", "reference")
CELL_DESCRIPTOR_NAMES = (
    "phase_center_sin",
    "phase_center_cos",
    "phase_slope",
    "phase_curvature",
    "rssi_center",
    "rssi_slope",
    "rssi_curvature",
    "rssi_shape_rmse",
)
QUALITY_DESCRIPTOR_NAMES = (
    "available_fraction",
    "valid_fraction_median",
    "valid_fraction_q10",
    "log10_condition_median",
    "signal_reconstruction_rmse_median",
)


@dataclass(frozen=True)
class SessionFeatureTable:
    session_ids: tuple[str, ...]
    physical: np.ndarray
    physical_names: tuple[str, ...]
    quality: np.ndarray
    quality_names: tuple[str, ...]
    raw_mask: np.ndarray
    raw_mask_names: tuple[str, ...]


def curve_descriptors(
    coefficients: np.ndarray,
    *,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """Return eight wrap-safe phase/RSSI descriptors for one spline cell."""

    values = np.asarray(coefficients, dtype=np.float64)
    design = (
        build_cubic_bspline_basis()
        if basis is None
        else np.asarray(basis, dtype=np.float64)
    )
    if (
        values.shape != (COEFFICIENT_CHANNEL_COUNT, COEFFICIENT_COUNT)
        or design.shape != (50, COEFFICIENT_COUNT)
        or not np.isfinite(values).all()
        or not np.isfinite(design).all()
    ):
        raise ValueError("curve descriptor inputs violate the spline contract")

    phase_sin = design @ values[0]
    phase_cos = design @ values[1]
    rssi = design @ values[2]
    phase = np.unwrap(np.arctan2(phase_sin, phase_cos))
    frequency = np.linspace(-1.0, 1.0, design.shape[0], dtype=np.float64)
    quadratic = np.column_stack(
        (np.ones_like(frequency), frequency, np.square(frequency))
    )
    phase_fit = np.linalg.lstsq(quadratic, phase, rcond=None)[0]
    rssi_fit = np.linalg.lstsq(quadratic, rssi, rcond=None)[0]
    rssi_residual = rssi - quadratic @ rssi_fit
    result = np.asarray(
        (
            np.sin(phase_fit[0]),
            np.cos(phase_fit[0]),
            phase_fit[1],
            phase_fit[2],
            rssi_fit[0],
            rssi_fit[1],
            rssi_fit[2],
            np.sqrt(np.mean(np.square(rssi_residual))),
        ),
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ValueError("curve descriptors are non-finite")
    return result


def _curve_descriptor_tensor(extracted: SpectrumCoefficients) -> np.ndarray:
    coefficients = np.asarray(extracted.coefficients, dtype=np.float64)
    availability = np.asarray(extracted.availability) > 0
    if (
        coefficients.ndim != 5
        or coefficients.shape[1:] != (3, 3, 7, 8)
        or availability.shape != coefficients.shape[:3]
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("session physical coefficient contract failed")
    basis = build_cubic_bspline_basis()
    curves = np.einsum(
        "fc,wbskc->wbskf",
        basis,
        coefficients,
        optimize=True,
    )
    phase = np.unwrap(np.arctan2(curves[..., 0, :], curves[..., 1, :]), axis=-1)
    rssi = curves[..., 2, :]
    frequency = np.linspace(-1.0, 1.0, 50, dtype=np.float64)
    quadratic = np.column_stack(
        (np.ones_like(frequency), frequency, np.square(frequency))
    )
    projector = np.linalg.pinv(quadratic)
    phase_fit = np.einsum("cf,wbsf->wbsc", projector, phase, optimize=True)
    rssi_fit = np.einsum("cf,wbsf->wbsc", projector, rssi, optimize=True)
    rssi_reconstructed = np.einsum(
        "fc,wbsc->wbsf",
        quadratic,
        rssi_fit,
        optimize=True,
    )
    rssi_rmse = np.sqrt(
        np.mean(np.square(rssi - rssi_reconstructed), axis=-1)
    )
    output = np.stack(
        (
            np.sin(phase_fit[..., 0]),
            np.cos(phase_fit[..., 0]),
            phase_fit[..., 1],
            phase_fit[..., 2],
            rssi_fit[..., 0],
            rssi_fit[..., 1],
            rssi_fit[..., 2],
            rssi_rmse,
        ),
        axis=-1,
    )
    output[~availability] = np.nan
    return output


def _late_minus_early(descriptors: np.ndarray) -> np.ndarray:
    early = descriptors[:, :, 1]
    late = descriptors[:, :, 2]
    available = np.isfinite(early).all(axis=-1) & np.isfinite(late).all(axis=-1)
    result = late - early
    early_angle = np.arctan2(early[..., 0], early[..., 1])
    late_angle = np.arctan2(late[..., 0], late[..., 1])
    difference = np.arctan2(
        np.sin(late_angle - early_angle),
        np.cos(late_angle - early_angle),
    )
    result[..., 0] = np.sin(difference)
    result[..., 1] = np.cos(difference)
    result[~available] = np.nan
    return result


def _physical_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{branch}__{block}__{descriptor}"
        for branch in BRANCH_NAMES
        for block in ("full", "late_minus_early")
        for descriptor in CELL_DESCRIPTOR_NAMES
    )


def _quality_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{branch}__quality__{descriptor}"
        for branch in BRANCH_NAMES
        for descriptor in QUALITY_DESCRIPTOR_NAMES
    )


def _raw_mask_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{branch}__slot_{slot}__freq_{frequency:02d}__mask"
        for branch in BRANCH_NAMES
        for slot in range(3)
        for frequency in range(50)
    )


def build_session_feature_table(
    extracted: SpectrumCoefficients,
    mask: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
) -> SessionFeatureTable:
    """Aggregate P2 spectra and canonical masks to one independent session row."""

    observed = np.asarray(mask)
    window_count = extracted.coefficients.shape[0]
    if (
        observed.shape != (window_count, 3, 3, 50)
        or observed.dtype != np.dtype("bool")
        or len(metadata) != window_count
    ):
        raise ValueError("session feature inputs are not window-aligned")
    sessions_by_window = tuple(str(row.get("session_id", "")) for row in metadata)
    window_ids = tuple(str(row.get("window_id", "")) for row in metadata)
    if (
        any(not value for value in sessions_by_window)
        or any(not value for value in window_ids)
        or len(set(window_ids)) != len(window_ids)
    ):
        raise ValueError("session feature metadata identity is invalid")
    session_ids = tuple(sorted(set(sessions_by_window)))
    descriptors = _curve_descriptor_tensor(extracted)
    full = descriptors[:, :, 0]
    delta = _late_minus_early(descriptors)
    window_physical = np.concatenate((full, delta), axis=-1)
    if window_physical.shape != (window_count, 3, 16):
        raise RuntimeError("physical window feature shape is invalid")

    physical_rows: list[np.ndarray] = []
    quality_rows: list[np.ndarray] = []
    raw_mask_rows: list[np.ndarray] = []
    session_array = np.asarray(sessions_by_window, dtype=object)
    availability = np.asarray(extracted.availability) > 0
    valid_fraction = np.asarray(extracted.valid_fraction, dtype=np.float64)
    condition = np.asarray(extracted.condition, dtype=np.float64)
    rmse = np.asarray(extracted.reconstruction_rmse, dtype=np.float64)
    if (
        availability.shape != (window_count, 3, 3)
        or valid_fraction.shape != availability.shape
        or condition.shape != availability.shape
        or rmse.shape != availability.shape + (7,)
    ):
        raise ValueError("session quality arrays violate the P2 contract")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for session in session_ids:
            indices = np.flatnonzero(session_array == session)
            physical_rows.append(
                np.nanmedian(window_physical[indices], axis=0).reshape(-1)
            )
            branch_quality: list[float] = []
            for branch in range(3):
                cell_available = availability[indices, branch].reshape(-1)
                branch_quality.append(float(cell_available.mean()))
                if cell_available.any():
                    fraction = valid_fraction[indices, branch].reshape(-1)[
                        cell_available
                    ]
                    cell_condition = condition[indices, branch].reshape(-1)[
                        cell_available
                    ]
                    signal_rmse = np.mean(
                        rmse[indices, branch, :, :3],
                        axis=-1,
                    ).reshape(-1)[cell_available]
                    branch_quality.extend(
                        (
                            float(np.median(fraction)),
                            float(np.quantile(fraction, 0.10)),
                            float(np.median(np.log10(cell_condition))),
                            float(np.median(signal_rmse)),
                        )
                    )
                else:
                    branch_quality.extend((np.nan, np.nan, np.nan, np.nan))
            quality_rows.append(np.asarray(branch_quality, dtype=np.float64))
            raw_mask_rows.append(
                observed[indices].mean(axis=0, dtype=np.float64).reshape(-1)
            )

    table = SessionFeatureTable(
        session_ids=session_ids,
        physical=np.asarray(physical_rows, dtype=np.float64),
        physical_names=_physical_feature_names(),
        quality=np.asarray(quality_rows, dtype=np.float64),
        quality_names=_quality_feature_names(),
        raw_mask=np.asarray(raw_mask_rows, dtype=np.float64),
        raw_mask_names=_raw_mask_feature_names(),
    )
    if (
        table.physical.shape != (len(session_ids), 48)
        or table.quality.shape != (len(session_ids), 15)
        or table.raw_mask.shape != (len(session_ids), 450)
    ):
        raise RuntimeError("session feature table dimensions are invalid")
    return table


def candidate_views(
    table: SessionFeatureTable,
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    """Return the nine predeclared source-ablation matrices."""

    physical = np.asarray(table.physical, dtype=np.float64)
    quality = np.asarray(table.quality, dtype=np.float64)
    raw_mask = np.asarray(table.raw_mask, dtype=np.float64)
    if (
        physical.shape != (len(table.session_ids), 48)
        or quality.shape != (len(table.session_ids), 15)
        or raw_mask.shape != (len(table.session_ids), 450)
        or len(table.physical_names) != 48
        or len(table.quality_names) != 15
        or len(table.raw_mask_names) != 450
    ):
        raise ValueError("candidate view table violates the feature registry")
    phase_indices = np.asarray(
        [
            branch * 16 + offset
            for branch in range(3)
            for offset in (*range(0, 4), *range(8, 12))
        ],
        dtype=np.int64,
    )
    rssi_indices = np.asarray(
        [
            branch * 16 + offset
            for branch in range(3)
            for offset in (*range(4, 8), *range(12, 16))
        ],
        dtype=np.int64,
    )

    def physical_slice(start: int, stop: int) -> tuple[np.ndarray, tuple[str, ...]]:
        return physical[:, start:stop], table.physical_names[start:stop]

    return {
        "raw_mask_only": (raw_mask, table.raw_mask_names),
        "coarse_quality_only": (quality, table.quality_names),
        "reference_only": physical_slice(32, 48),
        "sensing_only": physical_slice(16, 32),
        "d4_only": physical_slice(0, 16),
        "phase_only": (
            physical[:, phase_indices],
            tuple(table.physical_names[index] for index in phase_indices),
        ),
        "rssi_only": (
            physical[:, rssi_indices],
            tuple(table.physical_names[index] for index in rssi_indices),
        ),
        "physical_signal_only": (physical, table.physical_names),
        "physical_signal_plus_quality": (
            np.column_stack((physical, quality)),
            table.physical_names + table.quality_names,
        ),
    }
