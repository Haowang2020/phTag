"""Circular RFID transforms and masked per-window feature tensors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


FEATURE_COUNT = 7
BRANCH_D4 = 0
BRANCH_SENSING = 1
BRANCH_REFERENCE = 2
CANONICAL_FREQUENCY_GRID = tuple(np.round(np.arange(902.75, 927.25 + 0.25, 0.5), 2))
_RESULTANT_TOLERANCE = 1e-12


def wrap_phase_deg(value: float) -> float:
    """Wrap an angle to [-180, 180), preserving circular rather than linear geometry."""

    return float((value + 180.0) % 360.0 - 180.0)


def phase_phasor(phase_deg: float) -> complex:
    """Construct a unit phasor for scalar/unit-test angle utilities only."""

    return complex(np.exp(1j * np.deg2rad(float(phase_deg))))


def phasor_from_sin_cos(sine: float, cosine: float) -> complex:
    """Normalize a deployment-safe sin/cos observation into a phasor."""

    if not np.isfinite(sine) or not np.isfinite(cosine):
        return complex(np.nan, np.nan)
    value = complex(float(cosine), float(sine))
    magnitude = abs(value)
    return value / magnitude if magnitude > _RESULTANT_TOLERANCE else complex(np.nan, np.nan)


def circular_mean_phasor(values: Iterable[complex]) -> tuple[complex, float, int]:
    valid = np.asarray([value for value in values if np.isfinite(value.real) and np.isfinite(value.imag)])
    if valid.size == 0:
        return complex(np.nan, np.nan), float("nan"), 0
    mean = valid.mean()
    resultant = float(abs(mean))
    if resultant <= _RESULTANT_TOLERANCE:
        return complex(np.nan, np.nan), resultant, int(valid.size)
    return complex(mean / resultant), resultant, int(valid.size)


def circular_mean_degrees(values: Iterable[float]) -> float:
    phasor, _, count = circular_mean_phasor(phase_phasor(value) for value in values if np.isfinite(value))
    if count == 0:
        return float("nan")
    return float(np.rad2deg(np.angle(phasor)))


def complex_phase_difference(left: complex, right: complex) -> complex:
    """Return the unit phasor for ``left - right`` without wrapping degrees."""

    return left * np.conj(right)


def d4_phase_phasor(sensing_wet: complex, reference_wet: complex, sensing_dry: complex, reference_dry: complex) -> complex:
    """Four-factor physical cancellation expressed as a phasor product."""

    return sensing_wet * np.conj(reference_wet) * np.conj(sensing_dry) * reference_dry


def _valid(frame: pd.DataFrame, mask: str, columns: Sequence[str]) -> pd.DataFrame:
    usable = frame.loc[frame[mask].eq(1)].copy()
    return usable.dropna(subset=list(columns))


def _phasor_mean(frame: pd.DataFrame, prefix: str) -> tuple[complex, float, int]:
    return circular_mean_phasor(
        phasor_from_sin_cos(sine, cosine)
        for sine, cosine in zip(
            pd.to_numeric(frame[f"{prefix}_observed_phase_sin"], errors="coerce"),
            pd.to_numeric(frame[f"{prefix}_observed_phase_cos"], errors="coerce"),
            strict=True,
        )
    )


def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.median(values)) if values.size else float("nan")


def _gap_feature(frame: pd.DataFrame, column: str) -> float:
    value = _numeric_median(frame, column)
    return float(np.log1p(value)) if np.isfinite(value) and value >= 0.0 else float("nan")


def _real_scalar(value: object) -> float:
    return float(np.real(complex(value)))


def _canonical_frequency_grid(frequency_grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in frequency_grid)
    if len(values) != len(CANONICAL_FREQUENCY_GRID) or not np.allclose(values, CANONICAL_FREQUENCY_GRID, rtol=0.0, atol=1e-9):
        raise ValueError("tensor artifacts require the canonical 50-frequency grid")
    return values


def compute_dry_baselines(
    rows: pd.DataFrame, *, dry_stage: str | Sequence[str] = ("dry", "dry_baseline")
) -> pd.DataFrame:
    """Compute dry references strictly within ``session_id × frequency_mhz``."""

    required = {"session_id", "stage", "frequency_mhz", "sensing_read_valid_mask", "reference_read_valid_mask"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"dry baseline rows are missing columns: {sorted(missing)}")
    dry_stages = {dry_stage} if isinstance(dry_stage, str) else set(dry_stage)
    dry = rows.loc[rows["stage"].isin(dry_stages)]
    records: list[dict[str, object]] = []
    for (session_id, frequency), group in dry.groupby(["session_id", "frequency_mhz"], observed=True, sort=True):
        sensing = _valid(group, "sensing_read_valid_mask", ["sensing_observed_phase_sin", "sensing_observed_phase_cos", "sensing_observed_rssi_dbm"])
        reference = _valid(group, "reference_read_valid_mask", ["reference_observed_phase_sin", "reference_observed_phase_cos", "reference_observed_rssi_dbm"])
        sensing_phase, sensing_resultant, sensing_count = _phasor_mean(sensing, "sensing") if not sensing.empty else (complex(np.nan, np.nan), float("nan"), 0)
        reference_phase, reference_resultant, reference_count = _phasor_mean(reference, "reference") if not reference.empty else (complex(np.nan, np.nan), float("nan"), 0)
        if sensing_count == 0 and reference_count == 0:
            continue
        records.append(
            {
                "session_id": session_id, "frequency_mhz": float(frequency),
                "sensing_dry_phasor": sensing_phase, "reference_dry_phasor": reference_phase,
                "sensing_dry_resultant": sensing_resultant, "reference_dry_resultant": reference_resultant,
                "sensing_dry_rssi": _numeric_median(sensing, "sensing_observed_rssi_dbm"),
                "reference_dry_rssi": _numeric_median(reference, "reference_observed_rssi_dbm"),
                "sensing_dry_count": sensing_count, "reference_dry_count": reference_count,
            }
        )
    columns = [
        "session_id", "frequency_mhz", "sensing_dry_phasor", "reference_dry_phasor",
        "sensing_dry_resultant", "reference_dry_resultant", "sensing_dry_rssi",
        "reference_dry_rssi", "sensing_dry_count", "reference_dry_count",
    ]
    if not records:
        return pd.DataFrame(columns=columns).set_index(["session_id", "frequency_mhz"])
    return pd.DataFrame.from_records(records, columns=columns).set_index(["session_id", "frequency_mhz"]).sort_index()


def _branch_features(wet: pd.DataFrame, dry: pd.Series, branch: int) -> tuple[np.ndarray, bool]:
    if branch == BRANCH_D4:
        usable = _valid(
            wet,
            "pair_read_valid_mask",
            [
                "sensing_observed_phase_sin", "sensing_observed_phase_cos",
                "reference_observed_phase_sin", "reference_observed_phase_cos",
                "pair_differential_rssi_db", "pair_observed_inter_read_gap_s",
            ],
        )
        dry_sensing, dry_reference = complex(dry["sensing_dry_phasor"]), complex(dry["reference_dry_phasor"])
        dry_ready = all(np.isfinite(value.real) and np.isfinite(value.imag) for value in (dry_sensing, dry_reference))
        if usable.empty or not dry_ready:
            return np.zeros(FEATURE_COUNT, dtype=float), False
        wet_phase = [
            complex_phase_difference(phasor_from_sin_cos(ss, sc), phasor_from_sin_cos(rs, rc))
            for ss, sc, rs, rc in zip(
                usable["sensing_observed_phase_sin"], usable["sensing_observed_phase_cos"],
                usable["reference_observed_phase_sin"], usable["reference_observed_phase_cos"], strict=True,
            )
        ]
        wet_phasor, wet_resultant, count = circular_mean_phasor(wet_phase)
        delta = d4_phase_phasor(wet_phasor, complex(1), dry_sensing, dry_reference)
        rssi_delta = _numeric_median(usable, "pair_differential_rssi_db") - (
            _real_scalar(dry["sensing_dry_rssi"]) - _real_scalar(dry["reference_dry_rssi"])
        )
        dry_resultant = _real_scalar(dry["sensing_dry_resultant"]) * _real_scalar(dry["reference_dry_resultant"])
        gap = _gap_feature(usable, "pair_observed_inter_read_gap_s")
    else:
        prefix = "sensing" if branch == BRANCH_SENSING else "reference"
        usable = _valid(
            wet, f"{prefix}_read_valid_mask",
            [f"{prefix}_observed_phase_sin", f"{prefix}_observed_phase_cos", f"{prefix}_observed_rssi_dbm", f"{prefix}_observed_inter_read_gap_s"],
        )
        dry_phasor = complex(dry[f"{prefix}_dry_phasor"])
        if usable.empty or not (np.isfinite(dry_phasor.real) and np.isfinite(dry_phasor.imag)):
            return np.zeros(FEATURE_COUNT, dtype=float), False
        wet_phasor, wet_resultant, count = _phasor_mean(usable, prefix)
        delta = complex_phase_difference(wet_phasor, dry_phasor)
        rssi_delta = _numeric_median(usable, f"{prefix}_observed_rssi_dbm") - _real_scalar(dry[f"{prefix}_dry_rssi"])
        dry_resultant = _real_scalar(dry[f"{prefix}_dry_resultant"])
        gap = _gap_feature(usable, f"{prefix}_observed_inter_read_gap_s")
    values = np.array([delta.imag, delta.real, rssi_delta, wet_resultant, dry_resultant, math.log1p(count), gap], dtype=float)
    if not np.isfinite(values).all():
        return np.zeros(FEATURE_COUNT, dtype=float), False
    return values, True


def build_branch_tensor(wet_rows: pd.DataFrame, dry_baselines: pd.DataFrame, frequency_grid: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return D4/sensing/reference tensors and explicit boolean availability masks."""

    grid = _canonical_frequency_grid(frequency_grid)
    tensor = np.zeros((3, len(grid), FEATURE_COUNT), dtype=np.float32)
    mask = np.zeros((3, len(grid)), dtype=bool)
    if wet_rows.empty:
        return tensor, mask
    sessions = wet_rows["session_id"].dropna().unique()
    if len(sessions) != 1:
        raise ValueError("a branch tensor must contain exactly one session")
    session_id = sessions[0]
    for index, frequency in enumerate(grid):
        key = (session_id, frequency)
        if key not in dry_baselines.index:
            continue
        subset = wet_rows.loc[np.isclose(wet_rows["frequency_mhz"].to_numpy(dtype=float), frequency)]
        if subset.empty:
            continue
        dry = dry_baselines.loc[key]
        for branch in (BRANCH_D4, BRANCH_SENSING, BRANCH_REFERENCE):
            values, valid = _branch_features(subset, dry, branch)
            if valid:
                tensor[branch, index] = values
                mask[branch, index] = True
    return tensor, mask


def build_window_tensors(
    wet_rows: pd.DataFrame, dry_baselines: pd.DataFrame, frequency_grid: Sequence[float] = CANONICAL_FREQUENCY_GRID,
    *, start_s: float, duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build three physical branches, each divided into three temporal sub-bins."""

    grid = _canonical_frequency_grid(frequency_grid)
    if duration_s <= 0.0:
        raise ValueError("window duration must be positive")
    if "stage_time_s" not in wet_rows.columns:
        raise ValueError("window tensors require stage_time_s for temporal sub-binning")
    tensors = np.zeros((3, 3, len(grid), FEATURE_COUNT), dtype=np.float32)
    masks = np.zeros((3, 3, len(grid)), dtype=bool)
    bin_width = duration_s / 3.0
    time = pd.to_numeric(wet_rows["stage_time_s"], errors="coerce")
    for sub_bin in range(3):
        lower, upper = start_s + sub_bin * bin_width, start_s + (sub_bin + 1) * bin_width
        branch_tensor, branch_mask = build_branch_tensor(wet_rows.loc[(time >= lower) & (time < upper)], dry_baselines, grid)
        tensors[:, sub_bin], masks[:, sub_bin] = branch_tensor, branch_mask
    return tensors, masks
