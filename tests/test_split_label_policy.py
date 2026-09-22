from pathlib import Path

import pandas as pd
import pytest

from mst.data.split_label_policy import (
    apply_split_label_policies,
    malignant_from_split_label,
)


def test_malignant_from_split_label_policies():
    assert malignant_from_split_label("malignant") == 1
    assert malignant_from_split_label("benign") == 0
    assert malignant_from_split_label("high risk", high_risk_policy="malignant") == 1
    assert malignant_from_split_label("high-risk", high_risk_policy="benign") == 0
    assert malignant_from_split_label("high risk", high_risk_policy="exclude") is None
    assert malignant_from_split_label("dcis", dcis_policy="malignant") == 1
    assert malignant_from_split_label("dcis", dcis_policy="exclude") is None
    with pytest.raises(ValueError):
        malignant_from_split_label("high risk", high_risk_policy="drop")


def test_apply_split_label_policies_excludes_and_remaps():
    df = pd.DataFrame(
        {
            "UID": ["a", "b", "c", "d"],
            "label": ["benign", "high risk", "dcis", "malignant"],
            "Malignant": [0, 1, 1, 1],
        }
    )
    dropped = apply_split_label_policies(df, high_risk_policy="exclude", dcis_policy="malignant")
    assert list(dropped["UID"]) == ["a", "c", "d"]
    assert list(dropped["Malignant"]) == [0, 1, 1]

    remapped = apply_split_label_policies(df, high_risk_policy="benign", dcis_policy="benign")
    assert list(remapped["UID"]) == ["a", "b", "c", "d"]
    assert list(remapped["Malignant"]) == [0, 0, 0, 1]


def test_apply_split_label_policies_noop_without_label():
    df = pd.DataFrame({"UID": ["a"], "Malignant": [1]})
    out = apply_split_label_policies(df, high_risk_policy="exclude")
    assert list(out["UID"]) == ["a"]


def test_fold_filter_then_policy(tmp_path: Path):
    csv = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "UID": ["a_left", "b_left", "c_left"],
            "Fold": [0, 0, 0],
            "Split": ["train", "train", "val"],
            "label": ["benign", "high risk", "malignant"],
            "Malignant": [0, 1, 1],
        }
    ).to_csv(csv, index=False)
    df = pd.read_csv(csv)
    df = df[df["Fold"] == 0]
    kept = apply_split_label_policies(df, high_risk_policy="exclude")
    assert set(kept["UID"]) == {"a_left", "c_left"}
    all_rows = apply_split_label_policies(df, high_risk_policy="malignant")
    assert set(all_rows["UID"]) == {"a_left", "b_left", "c_left"}
    assert int(all_rows.loc[all_rows["UID"] == "b_left", "Malignant"].iloc[0]) == 1
