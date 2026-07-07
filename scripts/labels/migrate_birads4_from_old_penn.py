"""Move BI-RADS-4 (Suspicious) cases from old Penn EHR table into matches_birads4_all.

Reads ``lat_added_dummy_ehr_chat_no_birads4.csv``, finds rows whose
``studylevelassessment`` contains ``4: Suspicious``, removes them from the
old-Penn export, and appends them to the BI-RADS-4 workbook with column
alignment and laterality translation.

Laterality (old numeric ``laterality`` → new ``lat`` strings):
    1 → right
    2 → left
    0, 3 → both  (random side chosen later in new_penn_mapping)
    4, 5 → ``null`` string (missing lat; excluded from step3 until resolved)

Outputs (under repo paths by default; override with CLI):
    - ``table_utils/lat_added_dummy_ehr_chat_old_penn_v2.csv`` — old cohort minus suspicious
    - ``tables/matches_birads4_all_v2.xlsx`` — original BI-RADS-4 rows + migrated rows
    - ``table_utils/birads4_migrated_from_old_penn.csv`` — audit slice that was moved
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Default paths (repo-relative)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_TABLE = REPO_ROOT / "table_utils" / "lat_added_dummy_ehr_chat_no_birads4.csv"
DEFAULT_BIRADS4_TABLE = REPO_ROOT / "tables" / "matches_birads4_all.xlsx"
DEFAULT_OLD_OUT = REPO_ROOT / "table_utils" / "lat_added_dummy_ehr_chat_old_penn_v2.csv"
DEFAULT_BIRADS4_OUT = REPO_ROOT / "tables" / "matches_birads4_all_v2.xlsx"
DEFAULT_EXTRACT_AUDIT = REPO_ROOT / "table_utils" / "birads4_migrated_from_old_penn.csv"

OLD_ACC = "dummy_acc"
NEW_ACC = "newaccession"
OLD_LAT = "laterality"
NEW_LAT = "lat"
STUDY_ASSESSMENT = "studylevelassessment"

BIRADS4_SUSPICIOUS_MARKER = "4: suspicious"


def _is_birads4_suspicious(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return BIRADS4_SUSPICIOUS_MARKER in str(val).strip().lower()


def translate_laterality(val) -> str:
    """Map old Penn numeric laterality to BI-RADS-4 ``lat`` strings."""
    null_str = "null"
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return null_str
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return null_str
    try:
        code = int(float(s))
    except ValueError:
        return null_str
    mapping = {
        1: "right",
        2: "left",
        0: "both",
        3: "both",
        4: null_str,
        5: null_str,
    }
    return mapping.get(code, null_str)


def _old_col_map(df: pd.DataFrame) -> dict[str, str]:
    """Lowercase column name → actual column name in old table."""
    return {str(c).lower(): c for c in df.columns}


def _build_birads4_row(
    old_row: pd.Series,
    birads4_columns: list[str],
    old_by_lower: dict[str, str],
) -> dict:
    """One BI-RADS-4-shaped row from an old-Penn row."""
    out: dict = {col: pd.NA for col in birads4_columns}
    birads_lower = {str(c).lower(): c for c in birads4_columns}

    for low, bcol in birads_lower.items():
        if low == NEW_ACC:
            out[bcol] = str(old_row[old_by_lower[OLD_ACC]]).strip()
        elif low == NEW_LAT:
            out[bcol] = translate_laterality(old_row.get(old_by_lower.get(OLD_LAT, OLD_LAT)))
        elif low in old_by_lower:
            out[bcol] = old_row[old_by_lower[low]]

    return out


def migrate(
    old_table: Path,
    birads4_table: Path,
    old_out: Path,
    birads4_out: Path,
    extract_audit: Path,
) -> None:
    old_df = pd.read_csv(old_table, dtype=str)
    # Excel default NA list treats the literal string "null" as missing; keep it.
    birads_df = pd.read_excel(birads4_table, dtype=str, keep_default_na=False)

    if OLD_ACC not in old_df.columns:
        raise KeyError(f"Old table missing {OLD_ACC!r}: {old_table}")
    if STUDY_ASSESSMENT not in old_df.columns:
        raise KeyError(f"Old table missing {STUDY_ASSESSMENT!r}: {old_table}")
    if NEW_ACC not in birads_df.columns:
        raise KeyError(f"BI-RADS-4 table missing {NEW_ACC!r}: {birads4_table}")

    old_df[OLD_ACC] = old_df[OLD_ACC].astype(str).str.strip()
    birads_df[NEW_ACC] = birads_df[NEW_ACC].astype(str).str.strip()

    suspicious_mask = old_df[STUDY_ASSESSMENT].map(_is_birads4_suspicious)
    extract_df = old_df.loc[suspicious_mask].copy()
    keep_df = old_df.loc[~suspicious_mask].copy()

    existing_new = set(birads_df[NEW_ACC].dropna())
    extract_accs = set(extract_df[OLD_ACC])
    overlap = extract_accs & existing_new
    if overlap:
        print(
            f"WARNING: {len(overlap)} migrated accessions already in BI-RADS-4 table; "
            "they will be skipped on append (old table still drops them)."
        )
        extract_df = extract_df.loc[~extract_df[OLD_ACC].isin(overlap)].copy()

    old_by_lower = _old_col_map(old_df)
    new_rows = [
        _build_birads4_row(extract_df.loc[idx], list(birads_df.columns), old_by_lower)
        for idx in extract_df.index
    ]
    append_df = pd.DataFrame(new_rows, columns=birads_df.columns)
    merged_df = pd.concat([birads_df, append_df], ignore_index=True)

    if NEW_LAT in merged_df.columns:
        lat_col = [c for c in merged_df.columns if str(c).lower() == NEW_LAT][0]
        merged_df[lat_col] = merged_df[lat_col].astype(str).str.strip()
        merged_df[lat_col] = merged_df[lat_col].replace(
            {"": "null", "nan": "null", "NaN": "null", "<NA>": "null"}
        )

    # Audit file: old columns + translated lat
    audit = extract_df.copy()
    audit["lat_translated"] = audit[OLD_LAT].map(translate_laterality) if OLD_LAT in audit.columns else None

    old_out.parent.mkdir(parents=True, exist_ok=True)
    birads4_out.parent.mkdir(parents=True, exist_ok=True)
    extract_audit.parent.mkdir(parents=True, exist_ok=True)

    keep_df.to_csv(old_out, index=False)
    merged_df.to_excel(birads4_out, index=False)
    audit.to_csv(extract_audit, index=False)

    lat_counts = audit["lat_translated"].value_counts(dropna=False) if "lat_translated" in audit.columns else {}
    print(f"Old table:        {len(old_df)} rows  ({old_table})")
    print(f"  BI-RADS-4 suspicious extracted: {int(suspicious_mask.sum())}")
    print(f"  kept in old v2:                  {len(keep_df)}  -> {old_out}")
    print(f"BI-RADS-4 table:  {len(birads_df)} rows  ({birads4_table})")
    print(f"  appended:                        {len(append_df)}  (skipped overlap: {len(overlap)})")
    print(f"  merged total:                    {len(merged_df)}  -> {birads4_out}")
    if NEW_LAT in merged_df.columns:
        lat_col = [c for c in merged_df.columns if str(c).lower() == NEW_LAT][0]
        lat_s = merged_df[lat_col].astype(str).str.lower()
        print(
            f"  merged lat: right={(lat_s == 'right').sum()}, left={(lat_s == 'left').sum()}, "
            f"both={(lat_s == 'both').sum()}, null={(lat_s == 'null').sum()}"
        )
    print(f"Audit extract:    {extract_audit}")
    if len(lat_counts):
        print("  lat_translated:", lat_counts.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move 4: Suspicious cases from old Penn CSV into matches_birads4_all."
    )
    parser.add_argument("--old-table", type=Path, default=DEFAULT_OLD_TABLE)
    parser.add_argument("--birads4-table", type=Path, default=DEFAULT_BIRADS4_TABLE)
    parser.add_argument("--old-out", type=Path, default=DEFAULT_OLD_OUT)
    parser.add_argument("--birads4-out", type=Path, default=DEFAULT_BIRADS4_OUT)
    parser.add_argument("--extract-audit", type=Path, default=DEFAULT_EXTRACT_AUDIT)
    args = parser.parse_args()
    migrate(
        args.old_table,
        args.birads4_table,
        args.old_out,
        args.birads4_out,
        args.extract_audit,
    )


if __name__ == "__main__":
    main()
