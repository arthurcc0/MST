import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "preprocessing" / "new_penn"))
from make_joint_datasplit import _norm_id, make_joint_datasplit  # noqa: E402


def _write_splits(tmp: Path) -> tuple[Path, Path]:
    new = pd.DataFrame(
        {
            "dummy_acc": ["n1", "n1", "n2", "n2", "n3", "n3"],
            "UID": ["n1_left", "n1_left", "n2_left", "n2_left", "n3_left", "n3_left"],
            "side": ["left"] * 6,
            "lat": ["left"] * 6,
            "Malignant": [0, 0, 1, 1, 0, 0],
            "label": ["benign", "benign", "malignant", "malignant", "benign", "benign"],
            "MRN": ["M1", "M1", "M2", "M2", "M3", "M3"],
            "Fold": ["0", "1", "0", "1", "0", "1"],
            "Split": ["test", "train", "val", "test", "train", "val"],
        }
    )
    old = pd.DataFrame(
        {
            "dummy_acc": ["o1", "o2", "o3", "o1", "o2", "o3"],
            "UID": ["o1_left", "o2_left", "o3_left", "o1_left", "o2_left", "o3_left"],
            "side": ["left"] * 6,
            "lat": ["left"] * 6,
            "Malignant": [0, 1, 0, 0, 1, 0],
            "label": ["benign", "malignant", "benign", "benign", "malignant", "benign"],
            "PennChart_EpicPatientId": ["E1", "E2", "E3", "E1", "E2", "E3"],
            "MRN": ["M2", "OX", "M9", "M2", "OX", "M9"],
            "Fold": ["0", "0", "0", "1", "1", "1"],
            "Split": ["train", "val", "test", "train", "train", "train"],
        }
    )
    new_p = tmp / "new.csv"
    old_p = tmp / "old.csv"
    new.to_csv(new_p, index=False)
    old.to_csv(old_p, index=False)
    return new_p, old_p


def test_graft_preserves_new_test_and_drops_mrn_leaks(tmp_path: Path):
    new_p, old_p = _write_splits(tmp_path)
    out = tmp_path / "joint.csv"
    joint = make_joint_datasplit(
        new_split_csv=new_p,
        old_split_csv=old_p,
        output_csv=out,
        old_mapping_csv=None,
    )
    fold0 = joint[joint["Fold"].astype(str) == "0"]
    train0 = fold0[fold0["Split"] == "train"]
    test0 = fold0[fold0["Split"] == "test"]
    fold1 = joint[joint["Fold"].astype(str) == "1"]
    train1 = fold1[fold1["Split"] == "train"]
    assert set(test0["UID"]) == {"n1_left"}
    assert (test0["cohort"] == "new_penn").all()
    # Fold 0 uses old fold-0 train+val: o1 (train, leaks M2 vs new val) / o2 (val)
    assert "o1_left" not in set(train0["UID"])
    assert "o2_left" in set(train0["UID"])
    assert "o3_left" not in set(train0["UID"])
    assert "n3_left" in set(train0["UID"])
    # Fold 1 uses old fold-1 train: o1/o2/o3; o1 leaks vs new test M2
    assert "o1_left" not in set(train1["UID"])
    assert "o2_left" in set(train1["UID"])
    assert "o3_left" in set(train1["UID"])
    assert out.is_file()


def test_norm_id_strips_pandas_float_suffix():
    assert _norm_id("1144138.0") == "1144138"
    assert _norm_id("1144138") == "1144138"
    assert _norm_id(1144138.0) == "1144138"
    assert _norm_id("001.0") == "001"
    assert _norm_id("M2") == "M2"
    assert _norm_id("") == ""
    assert _norm_id("nan") == ""


def test_graft_matches_mrn_despite_float_suffix(tmp_path: Path):
    new = pd.DataFrame(
        {
            "dummy_acc": ["n1", "n2"],
            "UID": ["n1_left", "n2_left"],
            "side": ["left", "left"],
            "lat": ["left", "left"],
            "Malignant": [0, 1],
            "label": ["benign", "malignant"],
            "MRN": ["1144138.0", "2.0"],
            "Fold": ["0", "0"],
            "Split": ["test", "val"],
        }
    )
    old = pd.DataFrame(
        {
            "dummy_acc": ["o1", "o2"],
            "UID": ["o1_left", "o2_left"],
            "side": ["left", "left"],
            "lat": ["left", "left"],
            "Malignant": [0, 1],
            "label": ["benign", "malignant"],
            "PennChart_EpicPatientId": ["E1", "E2"],
            "MRN": ["1144138", "9"],
            "Fold": ["0", "0"],
            "Split": ["train", "train"],
        }
    )
    new_p = tmp_path / "new.csv"
    old_p = tmp_path / "old.csv"
    new.to_csv(new_p, index=False)
    old.to_csv(old_p, index=False)
    joint = make_joint_datasplit(
        new_split_csv=new_p,
        old_split_csv=old_p,
        output_csv=tmp_path / "joint.csv",
        old_mapping_csv=None,
    )
    train = joint[(joint["Fold"].astype(str) == "0") & (joint["Split"] == "train")]
    assert "o1_left" not in set(train["UID"])
    assert "o2_left" in set(train["UID"])
    grafted = joint[joint["UID"] == "o2_left"]
    assert set(grafted["MRN"]) == {"9"}
