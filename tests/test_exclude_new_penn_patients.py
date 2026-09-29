import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "preprocessing" / "new_penn"))
from step3b_exclude_new_penn_patients import exclude_new_penn_patients  # noqa: E402


def test_drops_shared_patients_and_keeps_folds(tmp_path: Path):
    old = pd.DataFrame(
        {
            "UID": ["a_left", "a_right", "b_left", "c_left"] * 2,
            "Malignant": [0, 1, 0, 0] * 2,
            "MRN": ["1144138", "1144138", "22", "M3"] * 2,
            "Fold": [0] * 4 + [1] * 4,
            "Split": ["train", "train", "val", "test", "test", "test", "train", "val"],
        }
    )
    old_csv = tmp_path / "old.csv"
    old.to_csv(old_csv, index=False)
    new_csv = tmp_path / "new.csv"
    pd.DataFrame({"MRN": ["1144138.0", None]}).to_csv(new_csv, index=False)
    holdout_csv = tmp_path / "holdout.csv"
    pd.DataFrame({"MRN": ["0000000022"]}).to_csv(holdout_csv, index=False)
    out_csv = tmp_path / "old_filtered.csv"

    kept = exclude_new_penn_patients(old_csv, out_csv, [new_csv, holdout_csv])

    assert set(kept["UID"]) == {"c_left"}
    written = pd.read_csv(out_csv, dtype=str)
    assert list(written.columns) == list(old.columns)
    assert written[["Fold", "Split"]].values.tolist() == [["0", "test"], ["1", "val"]]
    excluded = pd.read_csv(tmp_path / "old_filtered_excluded.csv", dtype=str)
    assert set(excluded["UID"]) == {"a_left", "a_right", "b_left"}
