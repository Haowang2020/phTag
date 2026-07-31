"""Raw RFID reads to the 16 D4 features used by the final model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_session_feature_table, candidate_views
from .preprocessing import (
    build_p1_multiresolution_tensor,
    extract_quality_weighted_spectrum,
)
from .windowing import SOURCE_COLUMNS, build_adaptive_windows


def extract_features(reads: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(reads.columns))
    if missing:
        raise ValueError("missing raw columns: " + ", ".join(missing))
    if reads.empty:
        raise ValueError("the raw input is empty")

    windows = []
    for session_id, rows in reads.groupby(
        reads["session_id"].astype(str),
        sort=True,
    ):
        accepted = [
            window
            for window in build_adaptive_windows(rows.loc[:, SOURCE_COLUMNS])
            if window.status == "accepted"
            and window.tensor is not None
            and window.mask is not None
        ]
        if not accepted:
            raise ValueError(f"{session_id}: no valid 3--8 s window")
        windows.extend(accepted)

    windows.sort(
        key=lambda item: (
            str(item.session_id),
            str(item.stage),
            float(item.start_s),
            float(item.duration_s),
        )
    )
    tensor = np.stack([item.tensor for item in windows]).astype(np.float32)
    mask = np.stack([item.mask for item in windows]).astype(bool)
    window_info = [
        {
            "window_id": (
                f"{item.session_id}::{item.stage}::"
                f"{item.start_s:.6f}::{item.duration_s:.6f}"
            ),
            "session_id": str(item.session_id),
        }
        for item in windows
    ]

    multiresolution, multiresolution_mask = build_p1_multiresolution_tensor(
        tensor,
        mask,
    )
    spectrum = extract_quality_weighted_spectrum(
        multiresolution,
        multiresolution_mask,
    )
    session_table = build_session_feature_table(spectrum, mask, window_info)
    values, names = candidate_views(session_table)["d4_only"]
    output = pd.DataFrame(values, columns=names)
    output.insert(0, "session_id", session_table.session_ids)
    return output
