"""Map canonical datasplit labels onto the binary Malignant flag at load time."""

from __future__ import annotations

import pandas as pd

SPLIT_LABEL_POLICIES = ("malignant", "benign", "exclude")


def normalize_split_label(label) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    return str(label).strip().lower().replace("-", " ")


def _policy_to_malignant(policy: str, *, name: str) -> int | None:
    p = str(policy).strip().lower()
    if p == "malignant":
        return 1
    if p == "benign":
        return 0
    if p == "exclude":
        return None
    raise ValueError(f"Invalid {name}={policy!r}; use {SPLIT_LABEL_POLICIES}.")


def malignant_from_split_label(
    label,
    high_risk_policy: str = "malignant",
    dcis_policy: str = "malignant",
) -> int | None:
    """Map a canonical table label to Malignant 0/1, or None to drop the row."""
    norm = normalize_split_label(label)
    if norm == "malignant":
        return 1
    if norm == "benign":
        return 0
    if norm == "high risk":
        return _policy_to_malignant(high_risk_policy, name="high_risk_policy")
    if norm == "dcis":
        return _policy_to_malignant(dcis_policy, name="dcis_policy")
    return 0


def apply_split_label_policies(
    df: pd.DataFrame,
    *,
    high_risk_policy: str = "malignant",
    dcis_policy: str = "malignant",
) -> pd.DataFrame:
    """Remap ``Malignant`` from ``label`` and drop rows whose policy is exclude.

    No-op when the frame has no ``label`` column (legacy splits).
    """
    if "label" not in df.columns:
        return df
    mapped = df["label"].map(
        lambda x: malignant_from_split_label(x, high_risk_policy, dcis_policy)
    )
    keep = mapped.notna()
    out = df.loc[keep].copy()
    out["Malignant"] = mapped.loc[keep].astype(int).to_numpy()
    return out
