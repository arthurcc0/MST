"""Aggregate side-level predict scores to accession level for the BRCA cohort.

Each accession may have ``<accession>_left`` and ``<accession>_right`` rows.
This script groups by ``dummy_acc`` and computes mean/max/median of ``NN_pred``
(and optional GT check — both sides should share the same label).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_accession_scores(
    predict_csv: Path,
    output_csv: Path,
    *,
    score_col: str = "NN_pred",
    method: str = "mean",
) -> pd.DataFrame:
    df = pd.read_csv(predict_csv)
    if "UID" not in df.columns:
        raise KeyError(f"Missing UID column in {predict_csv}")
    if score_col not in df.columns:
        raise KeyError(f"Missing {score_col!r} in {predict_csv}")

    df = df.copy()
    df["UID"] = df["UID"].astype(str).str.strip()
    if "dummy_acc" not in df.columns:
        df["dummy_acc"] = df["UID"].str.rsplit("_", n=1).str[0]

    agg_map = {
        "mean": "mean",
        "max": "max",
        "median": "median",
    }
    if method not in agg_map:
        raise ValueError(f"method must be one of {list(agg_map)}")
    agg_fn = agg_map[method]

    group_cols = ["dummy_acc"]
    if "Fold" in df.columns:
        group_cols.append("Fold")
    if "Split" in df.columns:
        group_cols.append("Split")

    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        n_sides=(score_col, "count"),
        accession_score=(score_col, agg_fn),
    ).reset_index()

    if "GT" in df.columns:
        gt_first = grouped["GT"].first().reset_index().rename(columns={"GT": "accession_GT"})
        gt_nuniq = grouped["GT"].nunique().reset_index(name="gt_n_unique")
        out = out.merge(gt_first, on=group_cols, how="left")
        out = out.merge(gt_nuniq, on=group_cols, how="left")
        mism = int((out["gt_n_unique"] > 1).sum())
        if mism:
            print(f"Warning: {mism} accession groups have inconsistent GT across sides.")
    elif "Malignant" in df.columns:
        mal_first = grouped["Malignant"].first().reset_index().rename(columns={"Malignant": "accession_GT"})
        out = out.merge(mal_first, on=group_cols, how="left")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} accession-level rows -> {output_csv} (method={method})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate left/right BRCA predict scores to accession level."
    )
    parser.add_argument("--predict-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--score-col", default="NN_pred")
    parser.add_argument(
        "--method",
        choices=("mean", "max", "median"),
        default="mean",
        help="How to combine left/right scores (default: mean).",
    )
    args = parser.parse_args()
    aggregate_accession_scores(
        args.predict_csv,
        args.output_csv,
        score_col=args.score_col,
        method=args.method,
    )


if __name__ == "__main__":
    main()
