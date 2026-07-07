"""Calibrate ROC-based suspicion bands on val and apply to test.

Four bands (increasing score / suspicion):
    Very low | Low | Intermediate | High

Default calibration uses three ROC operating points on pooled, deduped val
predictions. Boundaries are score thresholds at target false-positive rates
(FPR = 1 - specificity) on the val ROC curve:

    t_low   @ FPR 0.05  (95% specificity)  — Very low | Low
    t_mid   @ FPR 0.20  (80% specificity)  — Low | Intermediate
    t_high  @ FPR 0.40  (60% specificity)  — Intermediate | High

Higher scores fall into higher suspicion bands. Thresholds are fit on val only,
then applied unchanged to test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from aggregate_auc import aggregate, dedupe_uid_predictions, find_fold_results
from merge_new_penn_predict_results import (
    DEFAULT_DATASPLIT_CSV,
    DEFAULT_LABEL_TABLE_XLSX,
    DEFAULT_MAPPING_CSV,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TRAIN_RUNS_ROOT,
    _load_datasplit,
    _load_label_table,
    _load_mapping,
    merge_label_table,
    merge_predict_with_metadata,
)

BAND_LABELS = ["Very low", "Low", "Intermediate", "High"]
CLASS_COLORS = {
    "malignant": "#d62728",
    "benign": "#2ca02c",
    "high risk": "#1f77b4",
}
CLASS_ORDER = ["benign", "high risk", "malignant"]
DEFAULT_OUT_DIR = PROJECT_ROOT / "results-pretrained-oldpenn-on-newpenn"
BAND_CUTPOINT_COLORS = {
    "t_low": "#2ca02c",
    "t_mid": "#ff7f0e",
    "t_high": "#d62728",
}


def _canonical_class_label(label) -> str:
    text = str(label).strip().lower().replace("-", " ")
    text = " ".join(text.split())
    if text.replace(" ", "") == "highrisk":
        return "high risk"
    return text


def orient_scores(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=float)
    if roc_auc_score(y_true, y_score) < 0.5:
        return 1.0 - y_score
    return y_score


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Largest score threshold with FPR <= target_fpr on the val ROC."""
    y_score = orient_scores(y_true, y_score)
    fpr, _tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(thresholds[-1])
    idx = valid[np.argmax(fpr[valid])]
    return float(thresholds[idx])


def threshold_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float) -> float:
    """Smallest score threshold with TPR >= target_tpr on the val ROC."""
    y_score = orient_scores(y_true, y_score)
    _fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(tpr >= target_tpr)[0]
    if len(valid) == 0:
        return float(thresholds[0])
    idx = valid[np.argmin(tpr[valid])]
    return float(thresholds[idx])


def calibrate_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    mode: str = "fpr",
    fpr_cuts: tuple[float, float, float] = (0.05, 0.20, 0.40),
    tpr_cuts: tuple[float, float, float] = (0.50, 0.70, 0.90),
) -> dict:
    scores_flipped = bool(roc_auc_score(y_true, y_score) < 0.5)
    y_score = orient_scores(y_true, y_score)
    if mode == "fpr":
        raw = [threshold_at_fpr(y_true, y_score, f) for f in fpr_cuts]
        meta = {
            "mode": "fpr",
            "fpr_cuts": list(fpr_cuts),
            "specificity_cuts": [1.0 - f for f in fpr_cuts],
        }
    elif mode == "tpr":
        raw = [threshold_at_tpr(y_true, y_score, t) for t in tpr_cuts]
        meta = {"mode": "tpr", "tpr_cuts": list(tpr_cuts)}
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'fpr' or 'tpr'.")

    thresholds = sorted(raw)
    if len(set(thresholds)) < 3:
        # Degenerate ROC (very compressed scores): fall back to score quantiles.
        qs = np.quantile(y_score, [0.25, 0.50, 0.75])
        thresholds = sorted(float(q) for q in qs)
        meta["fallback"] = "quantile"
        meta["quantiles"] = [0.25, 0.50, 0.75]

    t_low, t_mid, t_high = thresholds
    return {
        **meta,
        "t_low": t_low,
        "t_mid": t_mid,
        "t_high": t_high,
        "scores_flipped": scores_flipped,
        "val_auc": float(roc_auc_score(y_true, y_score)),
        "val_n": int(len(y_true)),
        "val_pos": int(np.sum(y_true)),
    }


def scores_for_bands(scores: pd.Series, thresholds: dict) -> pd.Series:
    s = scores.astype(float)
    if thresholds.get("scores_flipped"):
        s = 1.0 - s
    return s


def assign_suspicion_bands(scores: pd.Series, thresholds: dict) -> pd.Series:
    bins = [-np.inf, thresholds["t_low"], thresholds["t_mid"], thresholds["t_high"], np.inf]
    return pd.cut(
        scores_for_bands(scores, thresholds),
        bins=bins,
        labels=BAND_LABELS,
        right=False,
        include_lowest=True,
    )


def _metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, thr: float, *, scores_flipped: bool = False
) -> dict:
    if scores_flipped:
        y_score = 1.0 - np.asarray(y_score, dtype=float)
    y_pred = y_score >= thr
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold": thr,
        "sensitivity": sens,
        "specificity": spec,
        "fpr": 1.0 - spec if not np.isnan(spec) else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _youden_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Score threshold maximizing Youden's J (TPR - FPR) on the ROC curve."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    _fpr, tpr, thrs = roc_curve(y_true, y_score)
    if len(thrs) == 0:
        return 0.5
    return float(thrs[int(np.argmax(tpr - _fpr))])


def print_confusion_matrix(
    df: pd.DataFrame,
    thresholds: dict,
    *,
    split_name: str = "test",
    score_threshold: float,
    threshold_source: str,
) -> None:
    """Print binary confusion matrix and sensitivity/specificity at a score cut."""
    y_true = df["GT"].astype(int).to_numpy()
    y_score = scores_for_bands(df["NN_pred"], thresholds).to_numpy(dtype=float)
    y_pred = y_score >= score_threshold
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(df)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tn + tp) / n if n else float("nan")

    cm_df = pd.DataFrame(
        cm,
        index=["True benign (GT=0)", "True malignant (GT=1)"],
        columns=["Pred benign (0)", "Pred malignant (1)"],
    )

    print()
    print(f"Confusion matrix ({split_name}, n={n}):")
    print(
        f"  Score threshold: {score_threshold:.4f} "
        f"(Youden optimal on {threshold_source})"
    )
    print()
    print(cm_df.to_string())
    print()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensitivity (TPR): {sens:.3f}  ({tp}/{tp + fn})")
    print(f"  Specificity (TNR): {spec:.3f}  ({tn}/{tn + fp})")
    print(f"  Accuracy:          {acc:.3f}")
    print()


def band_summary_table(df: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    rows = []
    for band in BAND_LABELS:
        sub = df[df["suspicion_band"] == band]
        if sub.empty:
            rows.append(
                {
                    "split": split_name,
                    "band": band,
                    "n": 0,
                    "n_malignant": 0,
                    "pct_malignant": float("nan"),
                    "score_min": float("nan"),
                    "score_median": float("nan"),
                    "score_max": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "split": split_name,
                "band": band,
                "n": len(sub),
                "n_malignant": int(sub["GT"].sum()),
                "pct_malignant": 100.0 * float(sub["GT"].mean()),
                "score_min": float(sub["NN_pred"].min()),
                "score_median": float(sub["NN_pred"].median()),
                "score_max": float(sub["NN_pred"].max()),
            }
        )
    return pd.DataFrame(rows)


def class_labels_from_df(df: pd.DataFrame) -> np.ndarray:
    if "mapping_label" in df.columns:
        labels = df["mapping_label"].astype(str).str.strip().str.lower()
    elif "split_label" in df.columns:
        labels = df["split_label"].astype(str).str.strip().str.lower()
    elif "label" in df.columns:
        labels = df["label"].astype(str).str.strip().str.lower()
    else:
        labels = df["GT"].map({0: "benign", 1: "malignant"})
    return np.array([_canonical_class_label(x) for x in labels])


def operating_points_per_case(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) on the ROC when threshold equals each case's score."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.zeros(len(y_true)), np.zeros(len(y_true))

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = np.array([np.sum(pos_mask & (y_score >= s)) / n_pos for s in y_score])
    fpr = np.array([np.sum(neg_mask & (y_score >= s)) / n_neg for s in y_score])
    return fpr, tpr


def _scatter_cases_on_roc(
    ax: plt.Axes,
    df: pd.DataFrame,
    thresholds: dict,
    *,
    jitter: float = 0.012,
    marker_size: float = 22,
    alpha: float = 0.7,
    seed: int = 0,
    legend: bool = True,
) -> None:
    y_true = df["GT"].to_numpy(dtype=int)
    y_score = scores_for_bands(df["NN_pred"], thresholds).to_numpy()
    class_labels = class_labels_from_df(df)
    fpr, tpr = operating_points_per_case(y_true, y_score)

    rng = np.random.default_rng(seed)
    fpr_j = np.clip(fpr + rng.normal(0, jitter, len(fpr)), 0.0, 1.0)
    tpr_j = np.clip(tpr + rng.normal(0, jitter, len(tpr)), 0.0, 1.0)

    for cls in CLASS_ORDER:
        mask = class_labels == cls
        if not np.any(mask):
            continue
        color = CLASS_COLORS.get(cls, "#7f7f7f")
        label = f"{cls} (n={int(mask.sum())})" if legend else None
        ax.scatter(
            fpr_j[mask],
            tpr_j[mask],
            s=marker_size,
            c=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.4,
            label=label,
            zorder=4,
        )


def roc_point_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr: float,
    *,
    scores_flipped: bool = False,
) -> tuple[float, float]:
    m = _metrics_at_threshold(
        y_true, y_score, thr, scores_flipped=scores_flipped
    )
    return float(m["fpr"]), float(m["sensitivity"])


def plot_roc_with_suspicion_bands(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: dict,
    summary: pd.DataFrame,
    out_path: Path,
    *,
    cohort: str | None = None,
    show_test_roc: bool = True,
) -> None:
    """Plot val ROC with band cutpoint markers and val band summary annotation."""
    fig, ax = plt.subplots(figsize=(8, 8))

    y_val = val_df["GT"].to_numpy()
    s_val = scores_for_bands(val_df["NN_pred"], thresholds).to_numpy()
    fpr, tpr, _ = roc_curve(y_val, s_val)
    auc_val = auc(fpr, tpr)
    ax.plot(
        fpr,
        tpr,
        color="C0",
        lw=1.5,
        alpha=0.45,
        linestyle="--",
        label=f"Val ROC (AUC = {auc_val:.2f})",
    )

    if show_test_roc and len(test_df):
        y_test = test_df["GT"].to_numpy()
        s_test = scores_for_bands(test_df["NN_pred"], thresholds).to_numpy()
        fpr_t, tpr_t, _ = roc_curve(y_test, s_test)
        auc_test = auc(fpr_t, tpr_t)
        ax.plot(
            fpr_t,
            tpr_t,
            color="C0",
            lw=2.5,
            label=f"Test ROC (AUC = {auc_test:.2f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="k", lw=1, alpha=0.6)

    flipped = thresholds.get("scores_flipped", False)
    y_raw = val_df["NN_pred"].to_numpy()
    cutpoints = [
        ("t_low", "Very low | Low", "#2ca02c", (10, 8)),
        ("t_mid", "Low | Intermediate", "#ff7f0e", (10, -14)),
        ("t_high", "Intermediate | High", "#d62728", (10, 8)),
    ]
    for key, band_label, color, offset in cutpoints:
        thr = thresholds[key]
        fpr_p, tpr_p = roc_point_at_threshold(
            y_val, y_raw, thr, scores_flipped=flipped
        )
        ax.plot(fpr_p, tpr_p, "o", color=color, markersize=9, zorder=5, markeredgecolor="k")
        ax.hlines(tpr_p, 0.0, fpr_p, colors=color, linestyles=":", alpha=0.55, lw=1.2)
        ax.vlines(fpr_p, 0.0, tpr_p, colors=color, linestyles=":", alpha=0.55, lw=1.2)
        ax.annotate(
            f"{band_label}\n"
            f"thr = {thr:.3g}\n"
            f"FPR = {fpr_p:.0%}  Sens = {tpr_p:.0%}\n"
            f"Spec = {1 - fpr_p:.0%}",
            (fpr_p, tpr_p),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor=color),
        )

    val_summary = summary[summary["split"] == "val"]
    summary_lines = ["Val suspicion bands (% malignant):"]
    for band in BAND_LABELS:
        row = val_summary[val_summary["band"] == band]
        if row.empty:
            continue
        r = row.iloc[0]
        summary_lines.append(
            f"  {band}: n={int(r['n'])}, {r['pct_malignant']:.1f}% mal"
        )
    if thresholds.get("mode") == "fpr" and thresholds.get("fpr_cuts"):
        fprs = thresholds["fpr_cuts"]
        summary_lines.append(
            f"FPR targets: {fprs[0]:.0%} / {fprs[1]:.0%} / {fprs[2]:.0%}"
        )
    summary_lines.append(
        f"Score cuts: {thresholds['t_low']:.3g} / "
        f"{thresholds['t_mid']:.3g} / {thresholds['t_high']:.3g}"
    )
    ax.text(
        0.03,
        0.03,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    title = "ROC with suspicion-band operating points (val calibration)"
    if cohort:
        title += f"\n{cohort}"
    ax.set_title(title, fontsize=11)
    ax.grid(color="#dddddd", alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"ROC plot saved to {out_path}")


def plot_roc_cases_by_class(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thresholds: dict,
    out_path: Path,
    *,
    cohort: str | None = None,
    show_test_roc: bool = True,
) -> None:
    """Plot ROC curves with each case at its operating point, colored by label class."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], linestyle="--", color="k", lw=1, alpha=0.6, zorder=1)

    if show_test_roc and len(test_df):
        _scatter_cases_on_roc(
            ax,
            test_df,
            thresholds,
            marker_size=24,
            alpha=0.75,
            jitter=0.012,
            seed=2,
            legend=True,
        )

    y_val = val_df["GT"].to_numpy()
    s_val = scores_for_bands(val_df["NN_pred"], thresholds).to_numpy()
    fpr, tpr, _ = roc_curve(y_val, s_val)
    auc_val = auc(fpr, tpr)
    ax.plot(
        fpr,
        tpr,
        color="0.35",
        lw=1.5,
        alpha=0.45,
        linestyle="--",
        label=f"Val ROC (AUC = {auc_val:.2f})",
        zorder=5,
    )

    if show_test_roc and len(test_df):
        y_test = test_df["GT"].to_numpy()
        s_test = scores_for_bands(test_df["NN_pred"], thresholds).to_numpy()
        fpr_t, tpr_t, _ = roc_curve(y_test, s_test)
        auc_test = auc(fpr_t, tpr_t)
        ax.plot(
            fpr_t,
            tpr_t,
            color="0.35",
            lw=2.5,
            label=f"Test ROC (AUC = {auc_test:.2f})",
            zorder=6,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    title = "ROC with test cases by class (val dashed, test solid + points)"
    if cohort:
        title += f"\n{cohort}"
    ax.set_title(title, fontsize=11)
    ax.grid(color="#dddddd", alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # note = (
    #     "Test cases only: (FPR, TPR) when threshold = that case's score.\n"
    #     "Small jitter for visibility. "
    #     "Colors: benign=green, high risk=blue, malignant=red."
    # )
    # ax.text(
    #     0.03,
    #     0.03,
    #     note,
    #     transform=ax.transAxes,
    #     fontsize=7.5,
    #     va="bottom",
    #     ha="left",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    # )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"ROC class-case plot saved to {out_path}")


def plot_score_distribution_by_class(
    df: pd.DataFrame,
    thresholds: dict,
    out_path: Path,
    *,
    split_name: str = "test",
    cohort: str | None = None,
    n_bins: int = 25,
) -> None:
    """Overlaid binned histogram of oriented scores, one series per label class."""
    scores = scores_for_bands(df["NN_pred"], thresholds).to_numpy(dtype=float)
    class_labels = class_labels_from_df(df)
    valid = np.isfinite(scores)
    scores = scores[valid]
    class_labels = class_labels[valid]

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(scores) == 0:
        ax.text(0.5, 0.5, "No scores to plot", ha="center", va="center", transform=ax.transAxes)
    else:
        bin_edges = np.linspace(float(scores.min()), float(scores.max()), n_bins + 1)
        legend_handles: list[Patch | Line2D] = []

        for cls in CLASS_ORDER:
            mask = class_labels == cls
            if not np.any(mask):
                continue
            color = CLASS_COLORS[cls]
            _, _, patches = ax.hist(
                scores[mask],
                bins=bin_edges,
                alpha=0.55,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=CLASS_ORDER.index(cls) + 1,
            )
            for patch in patches:
                patch.set_facecolor(color)
                patch.set_edgecolor("white")
                patch.set_alpha(0.55)
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="white",
                    alpha=0.55,
                    label=f"{cls} (n={int(mask.sum())})",
                ),
            )

        cutpoints = [
            ("t_low", "Very low | Low"),
            ("t_mid", "Low | Intermediate"),
            ("t_high", "Intermediate | High"),
        ]
        for key, band_label in cutpoints:
            thr = thresholds[key]
            color = BAND_CUTPOINT_COLORS[key]
            ax.axvline(
                thr,
                color=color,
                linestyle="--",
                lw=1.2,
                alpha=0.85,
                zorder=4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    lw=1.2,
                    label=f"{band_label} ({thr:.3g})",
                )
            )
        leg = ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
        leg.set_zorder(100)

    ax.set_xlabel("Score (oriented for suspicion bands)")
    ax.set_ylabel("Count")
    title = f"Score distribution by class ({split_name})"
    if cohort:
        title += f" — {cohort}"
    ax.set_title(title, fontsize=11)
    ax.grid(color="#dddddd", alpha=0.8, axis="y")
    ax.set_axisbelow(True)
    if len(scores) == 0:
        leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        leg.set_zorder(100)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Score distribution plot saved to {out_path}")


def stratified_band_summary(
    df: pd.DataFrame,
    *,
    split_name: str,
    strat_col: str,
) -> pd.DataFrame:
    if strat_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for strat_val, grp in df.groupby(strat_col, dropna=False):
        for band in BAND_LABELS:
            sub = grp[grp["suspicion_band"] == band]
            rows.append(
                {
                    "split": split_name,
                    "stratifier": strat_col,
                    "stratum": strat_val,
                    "band": band,
                    "n": len(sub),
                    "n_malignant": int(sub["GT"].sum()) if len(sub) else 0,
                    "pct_malignant": 100.0 * float(sub["GT"].mean()) if len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def load_predictions(
    *,
    cohort: str | None,
    split: str,
    folds: list[int],
    results_root: Path,
    train_runs_root: Path,
    csv_path: Path | None,
    dedupe_uid: bool,
) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    elif cohort:
        paths = find_fold_results(
            cohort=cohort,
            folds=folds,
            results_root=results_root,
            train_runs_root=train_runs_root,
            split=split,
        )
        df = aggregate(paths, plot="none", cohort=cohort, dedupe_uid=dedupe_uid)
    else:
        raise ValueError(f"Provide --{split}-csv or --cohort for {split} predictions.")

    df = df.copy()
    df["UID"] = df["UID"].astype(str).str.strip()
    df["GT"] = df["GT"].astype(int)
    df["NN_pred"] = df["NN_pred"].astype(float)
    df["predict_split"] = split
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit ROC-based suspicion-band thresholds on val and apply to test."
    )
    parser.add_argument("--cohort", type=str, default=None, help="Cohort tag in run folder names.")
    parser.add_argument("--val-csv", type=Path, default=None, help="Pooled/deduped val predictions CSV.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Pooled test predictions CSV.")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold indices when discovering results via --cohort.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Predict output root (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--train-runs-root",
        type=Path,
        default=DEFAULT_TRAIN_RUNS_ROOT,
        help=f"Training runs root (default: {DEFAULT_TRAIN_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--mode",
        choices=["fpr", "tpr"],
        default="fpr",
        help="ROC calibration mode: FPR cuts (default) or TPR cuts.",
    )
    parser.add_argument(
        "--fpr-cuts",
        type=float,
        nargs=3,
        default=[0.05, 0.20, 0.40],
        metavar=("FPR1", "FPR2", "FPR3"),
        help="Val ROC FPR targets for t_low/t_mid/t_high (default: 0.05 0.20 0.40).",
    )
    parser.add_argument(
        "--tpr-cuts",
        type=float,
        nargs=3,
        default=[0.50, 0.70, 0.90],
        metavar=("TPR1", "TPR2", "TPR3"),
        help="Val ROC TPR targets when --mode tpr (default: 0.50 0.70 0.90).",
    )
    parser.add_argument(
        "--no-dedupe-val",
        action="store_true",
        help="Do not dedupe duplicate val UIDs before calibration.",
    )
    parser.add_argument(
        "--merge-metadata",
        action="store_true",
        help=(
            "Attach mapping/datasplit metadata and label-table pathology columns "
            "(brca, PathCode, histology) to outputs."
        ),
    )
    parser.add_argument(
        "--merge-label-table",
        action="store_true",
        help=(
            "Attach brca/PathCode/histology from matches_birads4 label table only "
            "(no mapping/datasplit). Implied when --merge-metadata is set."
        ),
    )
    parser.add_argument(
        "--label-table-xlsx",
        type=Path,
        default=DEFAULT_LABEL_TABLE_XLSX,
        help=f"BI-RADS-4 label workbook for pathology merge (default: {DEFAULT_LABEL_TABLE_XLSX}).",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help=f"Mapping CSV (default: {DEFAULT_MAPPING_CSV}).",
    )
    parser.add_argument(
        "--datasplit-csv",
        type=Path,
        default=DEFAULT_DATASPLIT_CSV,
        help=f"Datasplit CSV (default: {DEFAULT_DATASPLIT_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save ROC plot with band operating points and val band summary.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="ROC PNG path (default: <out-dir>/roc_suspicion_bands_<cohort>.png).",
    )
    parser.add_argument(
        "--no-test-roc-on-plot",
        action="store_true",
        help="Omit dashed test ROC from the plot (val only).",
    )
    parser.add_argument(
        "--plot-cases-out",
        type=Path,
        default=None,
        help="Class-colored case ROC PNG (default: <out-dir>/roc_cases_by_class_<cohort>.png).",
    )
    parser.add_argument(
        "--no-plot-cases",
        action="store_true",
        help="Skip the class-colored case scatter ROC when using --plot.",
    )
    parser.add_argument(
        "--plot-scores-out",
        type=Path,
        default=None,
        help="Class score histogram PNG (default: <out-dir>/score_dist_by_class_<cohort>.png).",
    )
    parser.add_argument(
        "--no-plot-scores",
        action="store_true",
        help="Skip the class score histogram when using --plot.",
    )
    args = parser.parse_args()

    cohort = args.cohort.strip() if args.cohort else None
    tag = cohort or "predict"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = load_predictions(
        cohort=cohort,
        split="val",
        folds=args.folds,
        results_root=args.results_root,
        train_runs_root=args.train_runs_root,
        csv_path=args.val_csv,
        dedupe_uid=not args.no_dedupe_val,
    )
    if not args.no_dedupe_val and val_df["UID"].duplicated().any():
        print("Deduping val predictions:")
        val_df = dedupe_uid_predictions(val_df)

    test_df = load_predictions(
        cohort=cohort,
        split="test",
        folds=args.folds,
        results_root=args.results_root,
        train_runs_root=args.train_runs_root,
        csv_path=args.test_csv,
        dedupe_uid=False,
    )

    y_val = val_df["GT"].to_numpy()
    s_val = val_df["NN_pred"].to_numpy()
    thresholds = calibrate_thresholds(
        y_val,
        s_val,
        mode=args.mode,
        fpr_cuts=tuple(args.fpr_cuts),
        tpr_cuts=tuple(args.tpr_cuts),
    )

    print("Calibrated thresholds on val:")
    print(f"  mode: {thresholds.get('mode')}")
    print(f"  t_low  (Very low | Low):        {thresholds['t_low']:.6g}")
    print(f"  t_mid  (Low | Intermediate):    {thresholds['t_mid']:.6g}")
    print(f"  t_high (Intermediate | High):   {thresholds['t_high']:.6g}")
    print(f"  val AUC: {thresholds['val_auc']:.3f}  n={thresholds['val_n']}  pos={thresholds['val_pos']}")
    if thresholds.get("fallback"):
        print(f"  note: used {thresholds['fallback']} fallback (ROC cuts were degenerate)")

    for label, key in [("t_low", "t_low"), ("t_mid", "t_mid"), ("t_high", "t_high")]:
        m = _metrics_at_threshold(
            y_val,
            s_val,
            thresholds[key],
            scores_flipped=thresholds.get("scores_flipped", False),
        )
        print(
            f"  at {label}: sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"(FPR={m['fpr']:.3f})"
        )

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["suspicion_band"] = assign_suspicion_bands(val_df["NN_pred"], thresholds)
    test_df["suspicion_band"] = assign_suspicion_bands(test_df["NN_pred"], thresholds)

    merge_label = args.merge_label_table or args.merge_metadata
    if args.merge_metadata:
        mapping_df = _load_mapping(args.mapping_csv)
        datasplit_df = _load_datasplit(args.datasplit_csv)
        val_df = merge_predict_with_metadata(val_df, mapping_df, datasplit_df, deduped=True)
        test_df = merge_predict_with_metadata(test_df, mapping_df, datasplit_df, deduped=True)
    if merge_label:
        print(f"Merging label-table columns from {args.label_table_xlsx}:")
        label_df = _load_label_table(args.label_table_xlsx)
        val_df = merge_label_table(val_df, label_df)
        test_df = merge_label_table(test_df, label_df)

    thresholds_path = out_dir / f"suspicion_thresholds_{tag}.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **thresholds,
                "band_labels": BAND_LABELS,
                "cohort": cohort,
            },
            f,
            indent=2,
        )
    print(f"Thresholds saved to {thresholds_path}")

    val_out = out_dir / f"suspicion_bands_val_{tag}.csv"
    test_out = out_dir / f"suspicion_bands_test_{tag}.csv"
    pooled_out = out_dir / f"pooled_classification_{tag}.csv"
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)
    test_df.to_csv(pooled_out, index=False)
    print(f"Val bands saved to  {val_out}  (n={len(val_df)})")
    print(f"Test bands saved to {test_out}  (n={len(test_df)})")
    print(f"Pooled classification table: {pooled_out}  (n={len(test_df)})")

    summary_parts = [
        band_summary_table(val_df, split_name="val"),
        band_summary_table(test_df, split_name="test"),
    ]
    summary = pd.concat(summary_parts, ignore_index=True)
    summary_path = out_dir / f"suspicion_bands_summary_{tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to    {summary_path}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print_confusion_matrix(
        test_df,
        thresholds,
        split_name="test",
        score_threshold=_youden_optimal_threshold(
            y_val, scores_for_bands(val_df["NN_pred"], thresholds).to_numpy(dtype=float)
        ),
        threshold_source="val",
    )

    if args.plot:
        if "mapping_label" not in val_df.columns:
            mapping_for_plot = _load_mapping(args.mapping_csv)
            val_df = val_df.merge(mapping_for_plot, on="UID", how="left")
            test_df = test_df.merge(mapping_for_plot, on="UID", how="left")

        plot_path = args.plot_out or (out_dir / f"roc_suspicion_bands_{tag}.png")
        plot_roc_with_suspicion_bands(
            val_df,
            test_df,
            thresholds,
            summary,
            plot_path,
            cohort=cohort,
            show_test_roc=not args.no_test_roc_on_plot,
        )
        if not args.no_plot_cases:
            cases_path = args.plot_cases_out or (
                out_dir / f"roc_cases_by_class_{tag}.png"
            )
            plot_roc_cases_by_class(
                val_df,
                test_df,
                thresholds,
                cases_path,
                cohort=cohort,
                show_test_roc=not args.no_test_roc_on_plot,
            )
        if not args.no_plot_scores and len(test_df):
            scores_path = args.plot_scores_out or (
                out_dir / f"score_dist_by_class_{tag}.png"
            )
            plot_score_distribution_by_class(
                test_df,
                thresholds,
                scores_path,
                split_name="test",
                cohort=cohort,
            )

    if args.merge_metadata:
        strat_parts = []
        for split_name, frame in [("val", val_df), ("test", test_df)]:
            for col in ("mapping_label", "mapping_bc_prior"):
                if col in frame.columns:
                    strat_parts.append(
                        stratified_band_summary(frame, split_name=split_name, strat_col=col)
                    )
        if strat_parts:
            strat = pd.concat(strat_parts, ignore_index=True)
            strat_path = out_dir / f"suspicion_bands_stratified_{tag}.csv"
            strat.to_csv(strat_path, index=False)
            print(f"Stratified summary: {strat_path}")


if __name__ == "__main__":
    main()
