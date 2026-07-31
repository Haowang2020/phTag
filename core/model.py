"""The few lines needed to apply the saved eight-model Ridge ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _numbers(value: object) -> np.ndarray:
    if isinstance(value, str):
        return np.fromstring(value, sep=" ", dtype=float)
    return np.asarray(value, dtype=float)


def predict(features: pd.DataFrame, model_file: str | Path) -> pd.DataFrame:
    model = json.loads(Path(model_file).read_text(encoding="utf-8"))
    names = list(model["feature_names"])
    missing = sorted(set(names) - set(features.columns))
    if missing:
        raise ValueError("missing D4 features: " + ", ".join(missing))

    x = features[names].to_numpy(dtype=float)
    fold_predictions = []
    for saved in model["models"]:
        prep = saved["preprocessing"]
        ridge = saved["ridge"]
        median = _numbers(prep["medians"])
        mean = _numbers(prep["means"])
        scale = _numbers(prep["scales"])
        coefficient = _numbers(ridge["coefficient"])
        filled = np.where(np.isfinite(x), x, median)
        fold_predictions.append(
            ((filled - mean) / scale) @ coefficient + float(ridge["intercept"])
        )

    predicted_ph = np.clip(np.mean(fold_predictions, axis=0), 7.0, 10.0)
    return pd.DataFrame(
        {
            "session_id": features["session_id"].astype(str),
            "predicted_ph": predicted_ph,
        }
    )
