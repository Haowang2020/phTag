"""Leakage-resistant temporal window construction for training splits only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .physics import CANONICAL_FREQUENCY_GRID, build_window_tensors, compute_dry_baselines


WindowStatus = Literal["accepted", "insufficient_information"]
CALIBRATION_STAGES = frozenset({"dry", "dry_baseline"})
SOURCE_COLUMNS = [
    "session_id", "sample_index", "time_s", "stage", "stage_time_s", "frequency_mhz",
    "sensing_read_valid_mask", "reference_read_valid_mask", "pair_read_valid_mask",
    "sensing_observed_phase_sin", "sensing_observed_phase_cos", "sensing_observed_rssi_dbm",
    "reference_observed_phase_sin", "reference_observed_phase_cos", "reference_observed_rssi_dbm",
    "pair_differential_rssi_db", "sensing_observed_inter_read_gap_s",
    "reference_observed_inter_read_gap_s", "pair_observed_inter_read_gap_s",
]


@dataclass(frozen=True)
class WindowRecord:
    session_id: str
    stage: str
    start_s: float
    duration_s: float
    status: WindowStatus
    rows: pd.DataFrame
    tensor: np.ndarray | None = None
    mask: np.ndarray | None = None


def _pair_coverage(rows: pd.DataFrame) -> tuple[int, int]:
    valid = rows.loc[rows["pair_read_valid_mask"].eq(1)]
    return int(len(valid)), int(valid["frequency_mhz"].nunique())


def _timeline_end(times: np.ndarray) -> float:
    observed = times[np.isfinite(times)]
    return float(np.max(observed)) if observed.size else 0.0


def _validate_observed_frequencies(rows: pd.DataFrame) -> None:
    frequencies = pd.to_numeric(rows["frequency_mhz"], errors="coerce").dropna().to_numpy(dtype=float)
    if frequencies.size and not np.isclose(frequencies[:, None], np.asarray(CANONICAL_FREQUENCY_GRID)[None, :], atol=1e-9, rtol=0.0).any(axis=1).all():
        raise ValueError("tensor artifacts require canonical 50-frequency observations")


def _build_windows(rows: pd.DataFrame, *, adaptive: bool, materialize_tensors: bool = True) -> list[WindowRecord]:
    required = {"session_id", "stage", "stage_time_s", "frequency_mhz", "pair_read_valid_mask"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"window rows are missing columns: {sorted(missing)}")
    _validate_observed_frequencies(rows)
    dry_baselines = compute_dry_baselines(rows)
    records: list[WindowRecord] = []
    for (session_id, stage), group in rows.groupby(["session_id", "stage"], observed=True, sort=True):
        if stage in CALIBRATION_STAGES:
            continue
        group = group.loc[pd.to_numeric(group["stage_time_s"], errors="coerce").notna()].copy()
        if group.empty:
            continue
        group["stage_time_s"] = group["stage_time_s"].astype(float)
        available_end = _timeline_end(group["stage_time_s"].to_numpy())
        start = 0.0
        while available_end - start >= 3.0 - 1e-9:
            maximum = min(8.0 if adaptive else 3.0, available_end - start)
            maximum = np.floor((maximum + 1e-9) * 2.0) / 2.0
            if maximum < 3.0:
                break
            durations = np.arange(3.0, maximum + 1e-9, 0.5) if adaptive else np.array([3.0])
            selected_duration = float(durations[-1])
            status: WindowStatus = "insufficient_information"
            for duration in durations:
                candidate = group.loc[(group["stage_time_s"] >= start) & (group["stage_time_s"] < start + duration)]
                reads, frequencies = _pair_coverage(candidate)
                selected_duration = float(duration)
                if reads >= 12 and frequencies >= 10:
                    status = "accepted"
                    break
            selected = group.loc[(group["stage_time_s"] >= start) & (group["stage_time_s"] < start + selected_duration)].copy()
            if status == "accepted" and materialize_tensors:
                tensor, mask = build_window_tensors(
                    selected, dry_baselines, start_s=start, duration_s=selected_duration
                )
                records.append(WindowRecord(str(session_id), str(stage), float(start), selected_duration, status, selected, tensor, mask))
            else:
                records.append(WindowRecord(str(session_id), str(stage), float(start), selected_duration, status, selected))
            start += selected_duration
    return records


def build_adaptive_windows(rows: pd.DataFrame) -> list[WindowRecord]:
    """Create deterministic, non-overlapping prediction windows per session and stage."""

    return _build_windows(rows, adaptive=True)


def build_fixed_windows(rows: pd.DataFrame) -> list[WindowRecord]:
    """Strict three-second non-overlapping prediction-window ablation."""

    return _build_windows(rows, adaptive=False)


def summarize_adaptive_windows(rows: pd.DataFrame) -> dict[str, int]:
    """Count adaptive prediction windows without retaining tensor artifacts."""

    required = {"session_id", "stage", "stage_time_s", "frequency_mhz", "pair_read_valid_mask"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"window rows are missing columns: {sorted(missing)}")
    _validate_observed_frequencies(rows)
    accepted = extended = insufficient_information = 0
    for (_, stage), group in rows.groupby(["session_id", "stage"], observed=True, sort=True):
        if stage in CALIBRATION_STAGES:
            continue
        time = pd.to_numeric(group["stage_time_s"], errors="coerce")
        valid_time = time.notna()
        if not valid_time.any():
            continue
        time = time.loc[valid_time].to_numpy(dtype=float)
        pair_valid = group.loc[valid_time, "pair_read_valid_mask"].eq(1).to_numpy()
        frequency = pd.to_numeric(group.loc[valid_time, "frequency_mhz"], errors="coerce").to_numpy(dtype=float)
        available_end = _timeline_end(time)
        start = 0.0
        while available_end - start >= 3.0 - 1e-9:
            maximum = np.floor((min(8.0, available_end - start) + 1e-9) * 2.0) / 2.0
            if maximum < 3.0:
                break
            accepted_duration: float | None = None
            for duration in np.arange(3.0, maximum + 1e-9, 0.5):
                candidate = pair_valid & (time >= start) & (time < start + duration)
                if int(candidate.sum()) >= 12 and np.unique(frequency[candidate]).size >= 10:
                    accepted_duration = float(duration)
                    break
            if accepted_duration is None:
                selected_duration = float(maximum)
                insufficient_information += 1
            else:
                selected_duration = accepted_duration
                accepted += 1
                extended += selected_duration > 3.0
            start += selected_duration
    return {"accepted": accepted, "extended": extended, "insufficient_information": insufficient_information}


def align_window_arrays(windows: list[WindowRecord]) -> dict[str, np.ndarray | list[str]]:
    """Return independent accepted windows with equal total weight per session."""

    accepted = [window for window in windows if window.status == "accepted" and window.tensor is not None and window.mask is not None]
    if not accepted:
        return {
            "session_id": [],
            "tensor": np.empty((0, 3, 3, 50, 7), dtype=np.float32),
            "mask": np.empty((0, 3, 3, 50), dtype=bool),
            "weight": np.empty((0,), dtype=np.float32),
        }
    tensor = np.stack([window.tensor for window in accepted]).astype(np.float32, copy=False)
    mask = np.stack([window.mask for window in accepted]).astype(bool, copy=False)
    counts = pd.Series([window.session_id for window in accepted]).value_counts()
    weights = np.asarray([1.0 / counts[window.session_id] for window in accepted], dtype=np.float32)
    return {"session_id": [window.session_id for window in accepted], "tensor": tensor, "mask": mask, "weight": weights}


def pad_session_windows(windows: list[WindowRecord]) -> dict[str, np.ndarray | list[str]]:
    """Compatibility alias; windows are deliberately no longer session-padded."""

    return align_window_arrays(windows)
