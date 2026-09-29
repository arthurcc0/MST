"""Paired significance tests for two 5-fold CV models on the same test cases.

Both models must have been evaluated on the same folds (same UIDs per fold's
test set). Reports:

- Fold level (n = number of folds): paired t-test, the Nadeau–Bengio corrected
  t-test (accounts for overlapping training sets across CV folds), and the exact
  Wilcoxon signed-rank test. With 5 folds the smallest possible two-sided
  Wilcoxon p-value is 0.0625, so that test can never reach 0.05.
- Case level (n = every test case): DeLong test per fold, combined into a test
  of the mean-of-folds AUC difference, and a fold-stratified paired bootstrap
  CI for that difference. This uses the paired predictions on each case and is
  the one with real power.

    python scripts/analysis/compare_cv_models.py \
        --cohort-a birads4_..._lr1x10-5 --cohort-b joint_penn_oldF0_..._balCE \
        --label-a staged --label-b joint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
from aggregate_auc import find_fold_results  # noqa: E402

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results-pretrained-oldpenn-on-newpenn" / "PENN"
DEFAULT_TRAIN_RUNS_ROOT = PROJECT_ROOT / "runs" / "PENN"


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_paired(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> tuple[float, float, float]:
    """(auc_a, auc_b, var(auc_a - auc_b)) via DeLong's fast algorithm (Sun & Xu 2014)."""
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    aucs, v01, v10 = [], [], []
    for s in (score_a, score_b):
        tx, ty = _midrank(s[pos]), _midrank(s[~pos])
        tz = _midrank(np.concatenate([s[pos], s[~pos]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    sx = np.cov(np.vstack(v01))
    sy = np.cov(np.vstack(v10))
    cov = sx / m + sy / n
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return float(aucs[0]), float(aucs[1]), float(var_diff)


def load_paired(paths_a: list[Path], paths_b: list[Path]) -> list[pd.DataFrame]:
    folds = []
    for i, (pa, pb) in enumerate(zip(paths_a, paths_b)):
        a = pd.read_csv(pa)[["UID", "GT", "NN_pred"]].drop_duplicates("UID")
        b = pd.read_csv(pb)[["UID", "GT", "NN_pred"]].drop_duplicates("UID")
        m = a.merge(b, on="UID", suffixes=("_a", "_b"), how="inner")
        if len(m) != len(a) or len(m) != len(b):
            print(f"Warning fold {i}: {len(a)} vs {len(b)} cases, {len(m)} shared (using shared only).")
        if (m["GT_a"] != m["GT_b"]).any():
            raise ValueError(f"Fold {i}: GT differs between models for the same UID.")
        folds.append(pd.DataFrame({
            "UID": m["UID"], "y": m["GT_a"].astype(int),
            "a": m["NN_pred_a"].astype(float), "b": m["NN_pred_b"].astype(float), "fold": i,
        }))
    return folds


def fold_level(auc_a: np.ndarray, auc_b: np.ndarray, test_train_ratio: float) -> dict:
    d = auc_a - auc_b
    k = len(d)
    sd = d.std(ddof=1)
    t_plain = d.mean() / (sd / np.sqrt(k)) if sd > 0 else np.inf
    p_plain = 2 * stats.t.sf(abs(t_plain), df=k - 1)
    se_nb = np.sqrt((1 / k + test_train_ratio) * sd**2)
    t_nb = d.mean() / se_nb if se_nb > 0 else np.inf
    p_nb = 2 * stats.t.sf(abs(t_nb), df=k - 1)
    p_w = stats.wilcoxon(auc_a, auc_b, method="exact").pvalue
    return {"diff": d, "t_plain": t_plain, "p_plain": p_plain, "t_nb": t_nb,
            "p_nb": p_nb, "p_wilcoxon": p_w, "wins_a": int((d > 0).sum())}


def case_level(folds: list[pd.DataFrame], n_boot: int, seed: int) -> dict:
    k = len(folds)
    diffs, vars_ = [], []
    for f in folds:
        aa, ab, v = delong_paired(f["y"].to_numpy(), f["a"].to_numpy(), f["b"].to_numpy())
        diffs.append(aa - ab)
        vars_.append(v)
    mean_diff = float(np.mean(diffs))
    se = float(np.sqrt(np.sum(vars_)) / k)
    z = mean_diff / se
    p = float(2 * stats.norm.sf(abs(z)))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    arrays = [(f["y"].to_numpy(), f["a"].to_numpy(), f["b"].to_numpy()) for f in folds]
    for i in range(n_boot):
        ds = []
        for y, a, b in arrays:
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            ds.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        boots[i] = np.mean(ds)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))

    pool = pd.concat(folds, ignore_index=True)
    pa, pb, pv = delong_paired(pool["y"].to_numpy(), pool["a"].to_numpy(), pool["b"].to_numpy())
    p_pool = float(2 * stats.norm.sf(abs((pa - pb) / np.sqrt(pv))))
    return {"fold_diffs": diffs, "mean_diff": mean_diff, "se": se, "z": z, "p_delong": p,
            "ci": (lo, hi), "p_boot": p_boot, "pooled_auc_a": pa, "pooled_auc_b": pb,
            "p_pooled_delong": p_pool, "n_cases": len(pool)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-a", required=True)
    parser.add_argument("--cohort-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--train-runs-root", type=Path, default=DEFAULT_TRAIN_RUNS_ROOT)
    parser.add_argument("--test-train-ratio", type=float, default=None,
                        help="n_test/n_train for the Nadeau–Bengio correction (default: 1/(k-1)).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kw = dict(folds=args.folds, results_root=args.results_root,
              train_runs_root=args.train_runs_root, split=args.split)
    folds = load_paired(find_fold_results(args.cohort_a.strip(), **kw),
                        find_fold_results(args.cohort_b.strip(), **kw))
    auc_a = np.array([roc_auc_score(f["y"], f["a"]) for f in folds])
    auc_b = np.array([roc_auc_score(f["y"], f["b"]) for f in folds])
    ratio = args.test_train_ratio if args.test_train_ratio is not None else 1 / (len(folds) - 1)

    A, B = args.label_a, args.label_b
    print(f"\n{'fold':>4}  {'n':>4}  {A:>8}  {B:>8}  {A + '-' + B:>10}")
    for f, a, b in zip(folds, auc_a, auc_b):
        print(f"{int(f['fold'].iloc[0]):>4}  {len(f):>4}  {a:8.4f}  {b:8.4f}  {a - b:+10.4f}")
    print(f"mean        {auc_a.mean():8.4f}  {auc_b.mean():8.4f}  {auc_a.mean() - auc_b.mean():+10.4f}")

    fl = fold_level(auc_a, auc_b, ratio)
    print(f"\nFold level (n={len(folds)} folds; {A} better in {fl['wins_a']}/{len(folds)}):")
    print(f"  paired t-test                 t={fl['t_plain']:+.3f}  p={fl['p_plain']:.4f}")
    print(f"  Nadeau-Bengio corrected t     t={fl['t_nb']:+.3f}  p={fl['p_nb']:.4f}  (n_test/n_train={ratio:.3f})")
    print(f"  Wilcoxon signed-rank (exact)  p={fl['p_wilcoxon']:.4f}  (minimum possible with 5 folds: 0.0625)")

    cl = case_level(folds, args.bootstrap, args.seed)
    print(f"\nCase level (n={cl['n_cases']} test cases, paired on UID):")
    print(f"  mean fold AUC diff  {cl['mean_diff']:+.4f}  "
          f"95% bootstrap CI [{cl['ci'][0]:+.4f}, {cl['ci'][1]:+.4f}]  p={cl['p_boot']:.4f}")
    print(f"  DeLong (per fold, combined)   z={cl['z']:+.3f}  p={cl['p_delong']:.4f}  SE={cl['se']:.4f}")
    print(f"  DeLong on pooled scores       {A}={cl['pooled_auc_a']:.4f}  {B}={cl['pooled_auc_b']:.4f}  "
          f"p={cl['p_pooled_delong']:.4f}  (mixes fold calibrations; secondary)")
    detectable = 2.8 * cl["se"]
    print(f"\n  Smallest mean-AUC difference detectable with ~80% power at alpha=0.05: ~{detectable:.3f}")


if __name__ == "__main__":
    main()
