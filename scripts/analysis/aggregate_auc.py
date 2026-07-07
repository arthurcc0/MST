"""Aggregate per-fold main_predict results.csv files into CV AUC summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mst.utils.roc_curve import auc_bootstrapping

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results-duke-penn-training" / "runs" / "PENN"
DEFAULT_TRAIN_RUNS_ROOT = PROJECT_ROOT / "runs" / "PENN"


def _results_csv_name(split: str | None) -> str:
    if split in {"train", "val", "test"}:
        return f"results_{split}.csv"
    return "results.csv"


def _results_csv_for_run_name(
    run_name: str, results_root: Path, *, split: str | None = None
) -> Path:
    return results_root / run_name / _results_csv_name(split)


def results_dir_name(path_run: Path, *, fold: int | None = None) -> str:
    """Short stable directory name for predict outputs (avoids Windows MAX_PATH).

    PENN subtraction runs use the cohort tag after ``subtraction_`` in the run
    folder name, e.g. ``birads4_..._f2_reg`` instead of the full
    ``DinoClassifierSlice_2026_..._subtraction_birads4_...`` path.
    """
    name = path_run.name
    if "subtraction_" in name:
        name = name.split("subtraction_", 1)[1]
    if fold is not None and f"_f{fold}" not in name:
        name = f"{name}_f{fold}"
    return name


def _resolve_results_csv(
    run_path: Path,
    results_root: Path,
    *,
    split: str | None = None,
    fold: int | None = None,
) -> Path | None:
    for candidate in (results_dir_name(run_path, fold=fold), run_path.name):
        csv_path = _results_csv_for_run_name(candidate, results_root, split=split)
        if csv_path.is_file():
            return csv_path
    return None


def find_fold_results(
    cohort: str,
    folds: list[int],
    results_root: Path,
    train_runs_root: Path,
    *,
    split: str | None = None,
) -> list[Path]:
    """Pick latest training run folder per fold, map to predict results CSV."""
    paths: list[Path] = []
    for f in folds:
        matches = sorted(
            train_runs_root.glob(f"*{cohort}_f{f}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise FileNotFoundError(
                f"No run folder for fold {f} matching '*{cohort}_f{f}_*' under {train_runs_root}"
            )
        run_path = matches[0]
        csv_path = _resolve_results_csv(
            run_path, results_root, split=split, fold=f
        )
        if csv_path is None:
            short_name = results_dir_name(run_path, fold=f)
            raise FileNotFoundError(
                f"Missing predict output for fold {f}: "
                f"{_results_csv_for_run_name(short_name, results_root, split=split)} "
                f"(also checked legacy name {run_path.name})\n"
                f"(training run: {run_path})"
            )
        paths.append(csv_path)
    return paths


def _orient_scores(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    if roc_auc_score(y_true, y_score) < 0.5:
        return 1.0 - y_score
    return y_score


def _style_roc_axis(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color="k", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.grid(color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")


def _roc_with_std_band(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    label: str,
    color,
    bootstrapping: int,
    band_alpha: float = 0.2,
) -> float:
    y_score = _orient_scores(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)

    tprs, aucs, _, mean_fpr = auc_bootstrapping(
        y_true, y_score, bootstrapping=bootstrapping
    )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0, ddof=1)
    std_auc = np.std(aucs, ddof=1)

    ax.plot(
        fpr,
        tpr,
        color=color,
        lw=2,
        alpha=0.9,
        label=rf"{label} (AUC = {auc_val:.2f} $\pm$ {std_auc:.2f})",
    )
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color=color,
        alpha=band_alpha,
    )
    return float(auc_val)


def plot_pooled_roc(
    pool: pd.DataFrame,
    out_path: Path,
    bootstrapping: int = 1000,
    title: str = "Pooled ROC (all folds)",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = pool["GT"].to_numpy()
    y_score = pool["NN_pred"].to_numpy()
    _roc_with_std_band(
        ax,
        y_true,
        y_score,
        label="Pooled",
        color="C0",
        bootstrapping=bootstrapping,
        band_alpha=0.25,
    )
    _style_roc_axis(ax, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Pooled ROC plot saved to {out_path}")


def dedupe_uid_predictions(
    df: pd.DataFrame,
    *,
    strategy: str = "first",
    score_col: str = "NN_pred",
    label_col: str = "GT",
    score_tol: float = 1e-9,
) -> pd.DataFrame:
    """Collapse duplicate UIDs from pooled CV val rows to one row per case."""
    if "UID" not in df.columns:
        raise KeyError("Missing UID column for dedupe")

    df = df.copy()
    df["UID"] = df["UID"].astype(str).str.strip()
    n_rows = len(df)
    n_unique = df["UID"].nunique()
    if n_rows == n_unique:
        print(f"Dedupe: {n_rows} rows, all UIDs unique (no dedupe needed).")
        return df

    uid_counts = df.groupby("UID").size()
    n_dup_uids = int((uid_counts > 1).sum())
    print(
        f"Dedupe: {n_rows} rows -> {n_unique} unique UIDs "
        f"({n_dup_uids} UIDs appeared in multiple fold-val sets)."
    )

    if score_col in df.columns:
        spread = df.groupby("UID")[score_col].agg(["min", "max", "std", "count"])
        inconsistent = spread[spread["max"] - spread["min"] > score_tol]
        if len(inconsistent):
            print(
                f"  {len(inconsistent)} UIDs have differing {score_col} across folds "
                f"(max spread {float((inconsistent['max'] - inconsistent['min']).max()):.6g})."
            )
        else:
            print(f"  All duplicate UIDs have identical {score_col} across folds.")

    if strategy == "first":
        out = df.sort_values(["UID", "Fold"] if "Fold" in df.columns else ["UID"])
        out = out.drop_duplicates("UID", keep="first")
    elif strategy == "mean":
        group_cols = ["UID"]
        agg: dict[str, str | tuple[str, str]] = {}
        if score_col in df.columns:
            agg[score_col] = "mean"
        if label_col in df.columns:
            agg[label_col] = "first"
        if "NN" in df.columns:
            agg["NN"] = "first"
        if "Fold" in df.columns:
            agg["Fold"] = "first"
        if "predict_split" in df.columns:
            agg["predict_split"] = "first"
        if "results_csv" in df.columns:
            agg["results_csv"] = "first"
        out = df.groupby(group_cols, as_index=False).agg(agg)
    else:
        raise ValueError(f"Unknown dedupe strategy {strategy!r}; use 'first' or 'mean'.")

    if "Fold" in df.columns:
        fold_lists = (
            df.groupby("UID")["Fold"]
            .apply(lambda s: ",".join(str(int(x)) for x in sorted(s.unique())))
            .rename("source_folds")
        )
        out = out.merge(fold_lists, on="UID", how="left")
        out["n_source_folds"] = out["source_folds"].str.count(",") + 1

    return out.reset_index(drop=True)


def plot_folds_roc(
    parts: list[pd.DataFrame],
    out_path: Path,
    bootstrapping: int = 1000,
    title: str = "Per-fold ROC",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(parts), 1)))
    for i, df in enumerate(parts):
        _roc_with_std_band(
            ax,
            df["GT"].to_numpy(),
            df["NN_pred"].to_numpy(),
            label=f"Fold {i}",
            color=colors[i],
            bootstrapping=bootstrapping,
            band_alpha=0.12,
        )
    _style_roc_axis(ax, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Per-fold ROC plot saved to {out_path}")


def aggregate(
    results_csvs: list[Path],
    save_pooled: Path | None = None,
    plot: str = "none",
    plot_out: Path | None = None,
    bootstrapping: int = 1000,
    cohort: str | None = None,
    dedupe_uid: bool = False,
    dedupe_strategy: str = "first",
) -> pd.DataFrame:
    aucs: list[float] = []
    parts: list[pd.DataFrame] = []

    for i, csv_path in enumerate(results_csvs):
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        fold_auc = roc_auc_score(df["GT"], df["NN_pred"])
        aucs.append(fold_auc)
        df = df.copy()
        df["Fold"] = i
        df["results_csv"] = str(csv_path)
        parts.append(df)
        print(f"Fold {i}: n={len(df)}  pos={int(df['GT'].sum())}  AUC={fold_auc:.3f}  ({csv_path.name})")

    print(f"Mean AUC: {np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1):.3f}")
    pool = pd.concat(parts, ignore_index=True)
    print(f"Pooled AUC (all rows): {roc_auc_score(pool['GT'], pool['NN_pred']):.3f}  n={len(pool)}")

    pool_out = pool
    if dedupe_uid:
        pool_out = dedupe_uid_predictions(pool, strategy=dedupe_strategy)
        print(
            f"Pooled AUC (deduped): {roc_auc_score(pool_out['GT'], pool_out['NN_pred']):.3f}  "
            f"n={len(pool_out)}"
        )

    if save_pooled is not None:
        save_pooled = Path(save_pooled)
        save_pooled.parent.mkdir(parents=True, exist_ok=True)
        pool_out.to_csv(save_pooled, index=False)
        print(f"Pooled predictions saved to {save_pooled}")

    pool_for_plot = pool_out if dedupe_uid else pool

    tag = cohort or "cv"
    if plot in {"pooled", "folds", "both"}:
        if plot_out is not None:
            plot_out = Path(plot_out)
            if plot == "pooled":
                pooled_out = plot_out
                folds_out = None
            elif plot == "folds":
                pooled_out = None
                folds_out = plot_out
            else:
                pooled_out = plot_out.with_name(f"{plot_out.stem}_pooled{plot_out.suffix}")
                folds_out = plot_out.with_name(f"{plot_out.stem}_folds{plot_out.suffix}")
        else:
            pooled_out = PROJECT_ROOT / f"roc_pooled_{tag}.png" if plot in {"pooled", "both"} else None
            folds_out = PROJECT_ROOT / f"roc_folds_{tag}.png" if plot in {"folds", "both"} else None

        if pooled_out is not None:
            plot_pooled_roc(
                pool_for_plot,
                pooled_out,
                bootstrapping=bootstrapping,
                title=f"Pooled ROC — {tag}" + (" (deduped)" if dedupe_uid else ""),
            )
        if folds_out is not None:
            plot_folds_roc(
                parts,
                folds_out,
                bootstrapping=bootstrapping,
                title=f"Per-fold ROC — {tag}",
            )

    return pool_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate 5-fold (or custom) predict results.csv into CV AUC metrics."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        action="append",
        default=None,
        help="Explicit path(s) to results.csv (one per fold, in fold order). "
        "If omitted, use --cohort to auto-discover.",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort tag used in run folder names, e.g. new_penn_from_dino_freezebckbone_unfreeze2. "
        "Finds runs/PENN/*{cohort}_f{N}_* and matching results under results-duke-penn-training.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Predict split used when discovering results via --cohort "
        "(reads results_{split}.csv; default: results.csv).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold indices to include when using --cohort (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Directory containing PENN predict outputs (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--train-runs-root",
        type=Path,
        default=DEFAULT_TRAIN_RUNS_ROOT,
        help=f"Directory containing training run folders for --cohort discovery (default: {DEFAULT_TRAIN_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--save-pooled",
        type=Path,
        default=None,
        help="Optional path to write concatenated predictions (with Fold column).",
    )
    parser.add_argument(
        "--plot",
        choices=["none", "pooled", "folds", "both"],
        default="none",
        help="Plot pooled ROC, all fold ROCs on one figure, or both (default: none).",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Output PNG path. For --plot both, *_pooled.png and *_folds.png are derived from this stem.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for ROC std band (default: 1000).",
    )
    parser.add_argument(
        "--dedupe-uid",
        action="store_true",
        help="Collapse duplicate UIDs in the pooled table (typical for val: one row per case).",
    )
    parser.add_argument(
        "--dedupe-strategy",
        choices=["first", "mean"],
        default="first",
        help="How to combine duplicate UID rows (default: first).",
    )
    args = parser.parse_args()

    if args.results_csv:
        paths = args.results_csv
    elif args.cohort:
        paths = find_fold_results(
            cohort=args.cohort.strip(),
            folds=args.folds,
            results_root=args.results_root,
            train_runs_root=args.train_runs_root,
            split=args.split,
        )
    else:
        parser.error("Provide --results-csv and/or --cohort.")

    aggregate(
        paths,
        save_pooled=args.save_pooled,
        plot=args.plot,
        plot_out=args.plot_out,
        bootstrapping=args.bootstrap,
        cohort=args.cohort.strip() if args.cohort else None,
        dedupe_uid=args.dedupe_uid,
        dedupe_strategy=args.dedupe_strategy,
    )


if __name__ == "__main__":
    main()
