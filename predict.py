#!/usr/bin/env python3
"""Run the pHTag model on one or more raw reader CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.model import predict
from core.pipeline import extract_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reads = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in args.input],
        ignore_index=True,
    )
    features = extract_features(reads)
    result = predict(features, Path(__file__).with_name("model.json"))

    if args.output:
        result.to_csv(
            args.output,
            index=False,
            float_format="%.6f",
            lineterminator="\n",
        )
        print(f"saved {len(result)} prediction(s) to {args.output}")
    else:
        print(
            result.to_csv(
                index=False,
                float_format="%.6f",
                lineterminator="\n",
            ),
            end="",
        )


if __name__ == "__main__":
    main()
