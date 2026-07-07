"""Assign malignant / benign / high risk labels from MRI Narrative text.

Two modes:

1) **Batch + Cursor agent** (no API keys):

    python table_utils/define_label_from_narrative.py prepare-batch --batch-size 25
    # Agent reads tables/label_narrative_batch.json, writes label_narrative_batch_results.json
    python table_utils/define_label_from_narrative.py merge-batch
    # Or one step before agent classification:
    python table_utils/define_label_from_narrative.py next-batch --batch-size 25
    # Repeat until done, then:
    python table_utils/define_label_from_narrative.py report

2) **Direct LLM** (requires OPENAI_API_KEY or GOOGLE_API_KEY in .env):

    python table_utils/define_label_from_narrative.py run-llm --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TABLES_DIR = REPO_ROOT / "tables"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(REPO_ROOT / ".env")

DEFAULT_INPUT = TABLES_DIR / "matches_birads4_all_v2.xlsx"
DEFAULT_OUTPUT = TABLES_DIR / "matches_birads4_all_v2_with_narrative_labels.xlsx"
DEFAULT_MISMATCH_CSV = TABLES_DIR / "label_narrative_mismatches.csv"
DEFAULT_BATCH_JSON = TABLES_DIR / "label_narrative_batch.json"
DEFAULT_BATCH_RESULTS_JSON = TABLES_DIR / "label_narrative_batch_results.json"
DEFAULT_BATCH_PROMPT = TABLES_DIR / "label_narrative_batch_PROMPT.md"
DEFAULT_V3_OUTPUT = TABLES_DIR / "matches_birads4_all_v3.xlsx"

PATHOLOGY_MALIGNANT_COLS = (
    "behaviorcodeicdo3description",
    "PathologyClassification",
    "behaviorCodeIcdO3Description",
)

PATHOLOGY_BEHAVIOR_COL = "behaviorcodeicdo3description"

# ICD-O behavior text for in-situ / non-invasive carcinoma → high risk, not malignant.
PATHOLOGY_IN_SITU_PATTERNS = (
    r"carcinoma in situ",
    r"intraepithelial",
    r"noninfiltrating",
    r"non[- ]invasive",
)

ALLOWED_LABELS = frozenset({"benign", "malignant", "high risk"})

LABEL_RULES_MD = """\
# Label rules (return exactly one label per row)

| Label | Use when |
|-------|----------|
| **malignant** | Biopsy-proven malignancy or known cancer for the index finding on this study; invasive carcinoma/DCIS confirmed; recurrence confirmed malignant on pathology in the narrative. |
| **high risk** | Suspicious imaging **without** confirmed malignancy: BI-RADS 4/5, suspicious mass/NME/enhancement, biopsy recommended, malignancy cannot be excluded. |
| **benign** | Negative MRI / no suspicious enhancement; clearly benign finding only; BI-RADS 2/3 probably benign with no suspicious malignant-type finding; non-diagnostic/empty narrative (note in rationale). |

**Rules:** Prefer **high risk** over **benign** when a focal suspicious finding is described. Use **malignant** only when malignancy is confirmed or unequivocally known for the index finding.

**Output JSON** (`label_narrative_batch_results.json`): array of objects:

```json
[
  {"idx": 0, "label": "high risk", "rationale": "1-2 sentences citing the narrative."}
]
```

`label` must be exactly: `benign`, `malignant`, or `high risk`.
"""

PROMPT = """\
You are reviewing a breast MRI radiology narrative. Assign exactly ONE outcome label
for the **index breast MRI finding** that this exam is evaluating.

LABELS: malignant | high risk | benign

malignant = biopsy-proven or known cancer for index finding on this study.
high risk = suspicious (BI-RADS 4/5, biopsy recommended) but NOT confirmed malignant.
benign = negative / no suspicious enhancement / clearly benign only.

Prefer high risk over benign for suspicious focal findings.
Prefer malignant only when malignancy is confirmed.

Return label and a 1-3 sentence rationale citing key phrases.

Narrative:
{narrative}
"""


class LabelOutput(BaseModel):
    label: Literal["benign", "malignant", "high risk"]
    rationale: str = Field(description="Brief justification citing report phrases.")


def _normalize_label(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip().lower().replace("-", " ")
    if s in {"highrisk", "high  risk"}:
        return "high risk"
    return s


def _normalize_label_series(s: pd.Series) -> pd.Series:
    return s.map(_normalize_label)


def _is_empty_cell(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    s = str(val).strip().lower()
    return s in {"", "nan", "none"}


def _is_empty_narrative(narrative: str) -> bool:
    s = str(narrative).strip().lower()
    return s in {"", "nan", "none"}


_MALIGNANT_RE = [
    re.compile(p, re.I)
    for p in [
        r"biopsy[- ]proven\s+(malignan|invasive|idc|ilc|carcinoma|cancer)",
        r"biopsy proven\s+(malignan|invasive|idc|ilc|carcinoma|cancer)",
        r"pathology[- ]proven\s+(malignan|cancer|carcinoma)",
        r"known\s+(right|left\s+)?breast\s+cancer",
        r"known\s+cancer\s+measuring",
        r"known\s+malignancy",
        r"new\s+diagnosis\s+of\s+(right|left\s+)?breast\s+cancer",
        r"staging\s+evaluation\s+of\s+new\s+breast\s+cancer",
        r"recently\s+diagnosed\s+\w+\s+(invasive|idc|ilc|carcinoma)",
        r"newly\s+diagnosed\s+\w+\s+(invasive|idc|ilc|carcinoma)",
        r"invasive\s+ductal\s+carcinoma\s+and\s+ductal\s+carcinoma",
        r"invasive\s+lobular\s+carcinoma\s+and\s+ductal\s+carcinoma",
        r"confirmed\s+malignant\s+recurrence",
        r"recurrence\s+confirmed\s+malignant",
    ]
]

_HIGH_RISK_RE = [
    re.compile(p, re.I)
    for p in [
        r"biopsy\s+(is\s+)?recommended",
        r"mri[- ]guided\s+(core\s+)?biopsy",
        r"mr\s+guided\s+(core\s+)?biopsy",
        r"excisional\s+biopsy\s+(is\s+)?recommended",
        r"core\s+(needle\s+)?biopsy\s+(is\s+)?recommended",
        r"requires\s+tissue\s+diagnosis",
        r"suspicious\s+(mass|enhancement|finding|non[- ]mass|nme)",
        r"irregular(ly)?\s+(shaped\s+)?mass",
        r"delayed\s+phase\s+is\s+washout",
        r"washout\s+kinetics",
        r"restricted\s+diffusion",
        r"diffusion\s+restriction",
        r"cannot\s+be\s+excluded",
        r"concerning\s+for\s+malignancy",
        r"bloody\s+nipple\s+discharge",
        r"spontaneous\s+\w+\s+nipple\s+discharge",
        r"non[- ]mass\s+enhancement",
        r"nonmass\s+enhancement",
        r"clumped\s+non[- ]mass",
        r"segmental\s+non[- ]mass",
        r"segmental\s+nme",
        r"indeterminate",
        r"biopsy\s+of\s+(the\s+)?(right|left|both)",
        r"dominant\s+focus\s+of\s+enhancement",
        r"new\s+finding",
        r"interval\s+(increase|development|enlargement)",
    ]
]

_STRONG_BENIGN_RE = [
    re.compile(p, re.I)
    for p in [
        r"there\s+is\s+no\s+suspicious\s+enhancement",
        r"no\s+suspicious\s+enhancement",
        r"no\s+enhancement\s+suspicious\s+for\s+malignancy",
        r"no\s+dominant\s+suspicious",
        r"negative\s+for\s+malignancy",
        r"no\s+suspicious\s+finding",
        r"previously\s+biopsied\s+and\s+benign",
        r"biopsy\s+yielded.{0,80}benign",
        r"benign\s+(result|pathology|apocrine\s+cyst|fibroadenoma)",
        r"in\s+keeping\s+with\s+benignity",
        r"favored\s+to\s+represent\s+benign",
        r"likely\s+benign",
        r"six[- ]month\s+follow[- ]up\s+(is\s+)?recommended",
        r"short\s+term\s+follow[- ]up",
        r"stable\s+\d+\s*mm\s+enhancing\s+mass.{0,40}benign",
    ]
]


def classify_narrative_heuristic(narrative: str) -> tuple[str, str]:
    """Rule-based label for batch automation (prefer high risk over benign for suspicious findings)."""
    if _is_empty_narrative(narrative):
        return "benign", "Empty or non-diagnostic narrative; defaulted to benign."

    text = narrative.lower()
    findings = text.split("findings:", 1)[-1] if "findings:" in text else text

    for pat in _MALIGNANT_RE:
        m = pat.search(text)
        if m and not re.search(r"not\s+(proven|confirmed|pathology[- ]proven)", text[max(0, m.start() - 20):m.end() + 20], re.I):
            return "malignant", f"Narrative indicates confirmed malignancy ('{m.group(0).strip()}')."

    high_hits = [p.pattern for p in _HIGH_RISK_RE if p.search(findings)]
    benign_hits = [p.pattern for p in _STRONG_BENIGN_RE if p.search(findings)]

    if high_hits:
        return "high risk", "Suspicious imaging features without confirmed malignancy on this study."

    # Check whole-breast negative patterns on each side
    side_neg = re.findall(
        r"(right|left)\s+(there\s+is\s+no\s+suspicious\s+enhancement|no\s+suspicious\s+enhancement)",
        findings,
        re.I,
    )
    has_mass_or_nme = bool(
        re.search(r"\bmass\b|\bnon[- ]mass\b|\bnme\b|enhancing\s+mass", findings, re.I)
    )
    if len(side_neg) >= 2 and not has_mass_or_nme:
        return "benign", "Bilateral report with no suspicious enhancement on either side."

    if benign_hits and not has_mass_or_nme:
        return "benign", "Report describes benign or negative findings without dominant suspicious enhancement."

    if re.search(r"\bmass\b|\bnon[- ]mass\b|enhancing\s+mass|segmental\s+enhancement", findings, re.I):
        return "high risk", "Focal enhancing abnormality described without confirmed malignancy."

    if benign_hits:
        return "benign", "Predominantly benign/negative narrative without confirmed suspicious focal finding."

    return "benign", "No clear suspicious malignant-type finding identified; defaulted to benign."


def run_all_batches(
    input_path: Path,
    output_path: Path,
    *,
    narrative_col: str = "Narrative",
    label_col: str = "label",
    narrative_label_col: str = "label_narrative",
    rationale_col: str = "label_narrative_rationale",
    batch_size: int = 25,
    mismatch_csv: Path = DEFAULT_MISMATCH_CSV,
    use_heuristic: bool = True,
) -> None:
    """Classify all remaining rows in-process (no concurrent xlsx writes)."""
    df = _ensure_label_columns(
        _load_working_df(input_path, output_path),
        narrative_label_col=narrative_label_col,
        rationale_col=rationale_col,
    )
    batch_num = 0
    while True:
        eligible = _eligible_indices(
            df, narrative_col=narrative_col, narrative_label_col=narrative_label_col
        )
        if not eligible:
            break
        batch_idx = eligible[:batch_size]
        batch_num += 1
        for i in batch_idx:
            narrative = str(df.at[i, narrative_col])
            if use_heuristic:
                label, rationale = classify_narrative_heuristic(narrative)
            else:
                raise RuntimeError("run-all-batches requires use_heuristic=True (or use run-llm).")
            df.at[i, narrative_label_col] = label
            df.at[i, rationale_col] = rationale
        _write_table(df, output_path)
        remaining = len(
            _eligible_indices(df, narrative_col=narrative_col, narrative_label_col=narrative_label_col)
        )
        print(f"Batch {batch_num}: classified {len(batch_idx)} rows | remaining {remaining}")

    report(df, label_col=label_col, narrative_label_col=narrative_label_col, mismatch_csv=mismatch_csv)
    _write_table(df, output_path)


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Wrote {path}")


def _load_working_df(input_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.is_file():
        return _read_table(output_path)
    return _read_table(input_path)


def _ensure_label_columns(
    df: pd.DataFrame,
    *,
    narrative_label_col: str,
    rationale_col: str,
) -> pd.DataFrame:
    for col in (narrative_label_col, rationale_col):
        if col not in df.columns:
            df[col] = ""
    return df


def _eligible_indices(
    df: pd.DataFrame,
    *,
    narrative_col: str,
    narrative_label_col: str,
) -> list[int]:
    narrative = df[narrative_col].astype(str).str.strip()
    has_narr = (narrative != "") & (narrative.str.lower() != "nan")
    needs_label = df[narrative_label_col].map(_is_empty_cell)
    return [int(i) for i in df.index[has_narr & needs_label]]


def prepare_batch(
    input_path: Path,
    output_path: Path,
    *,
    batch_json: Path,
    batch_prompt_path: Path,
    narrative_col: str = "Narrative",
    label_col: str = "label",
    narrative_label_col: str = "label_narrative",
    batch_size: int = 25,
) -> None:
    df = _ensure_label_columns(
        _load_working_df(input_path, output_path),
        narrative_label_col=narrative_label_col,
        rationale_col="label_narrative_rationale",
    )
    eligible = _eligible_indices(
        df, narrative_col=narrative_col, narrative_label_col=narrative_label_col
    )
    batch_idx = eligible[:batch_size]

    batch = []
    for i in batch_idx:
        batch.append(
            {
                "idx": int(i),
                "current_label": _normalize_label(df.at[i, label_col]) if label_col in df.columns else "",
                "narrative": str(df.at[i, narrative_col]),
            }
        )

    batch_json = Path(batch_json)
    batch_json.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)

    batch_prompt_path = Path(batch_prompt_path)
    with open(batch_prompt_path, "w", encoding="utf-8") as f:
        f.write(LABEL_RULES_MD)
        f.write(f"\n\nBatch file: `{batch_json.name}` ({len(batch)} rows)\n")
        if batch:
            f.write(f"Index range: {batch[0]['idx']} .. {batch[-1]['idx']}\n")

    print(f"Eligible remaining: {len(eligible)}")
    print(f"Prepared batch: {len(batch)} rows -> {batch_json}")
    print(f"Agent instructions: {batch_prompt_path}")
    print()
    print("Next: ask the Cursor agent to classify the batch and write:")
    print(f"  {DEFAULT_BATCH_RESULTS_JSON}")


def merge_batch(
    input_path: Path,
    output_path: Path,
    *,
    batch_results_json: Path,
    narrative_label_col: str = "label_narrative",
    rationale_col: str = "label_narrative_rationale",
    mismatch_csv: Path = DEFAULT_MISMATCH_CSV,
    label_col: str = "label",
) -> None:
    results_path = Path(batch_results_json)
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing batch results: {results_path}")

    with open(results_path, encoding="utf-8-sig") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError(f"Expected JSON array in {results_path}")

    df = _ensure_label_columns(
        _load_working_df(input_path, output_path),
        narrative_label_col=narrative_label_col,
        rationale_col=rationale_col,
    )

    n_ok = n_bad = 0
    for row in results:
        if not isinstance(row, dict) or "idx" not in row:
            n_bad += 1
            continue
        idx = int(row["idx"])
        label = _normalize_label(row.get("label", ""))
        if label not in ALLOWED_LABELS:
            print(f"[warn] idx={idx}: invalid label {row.get('label')!r}; skipped")
            n_bad += 1
            continue
        df.at[idx, narrative_label_col] = label
        df.at[idx, rationale_col] = str(row.get("rationale", "")).strip()
        n_ok += 1

    _write_table(df, output_path)
    print(f"Merged {n_ok} rows from {results_path} (skipped/invalid: {n_bad})")

    remaining = len(
        _eligible_indices(df, narrative_col="Narrative", narrative_label_col=narrative_label_col)
    )
    print(f"Rows still needing label_narrative: {remaining}")

    if remaining == 0:
        report(
            df,
            label_col=label_col,
            narrative_label_col=narrative_label_col,
            mismatch_csv=mismatch_csv,
        )


def next_batch(
    input_path: Path,
    output_path: Path,
    *,
    batch_json: Path,
    batch_prompt_path: Path,
    batch_results_json: Path,
    narrative_col: str = "Narrative",
    label_col: str = "label",
    narrative_label_col: str = "label_narrative",
    batch_size: int = 25,
    mismatch_csv: Path = DEFAULT_MISMATCH_CSV,
) -> None:
    """Merge prior agent results, then export the next batch for classification."""
    if Path(batch_results_json).is_file():
        merge_batch(
            input_path,
            output_path,
            batch_results_json=batch_results_json,
            narrative_label_col=narrative_label_col,
            label_col=label_col,
            mismatch_csv=mismatch_csv,
        )
    else:
        print(f"No batch results at {batch_results_json}; skipping merge.")
    prepare_batch(
        input_path,
        output_path,
        batch_json=batch_json,
        batch_prompt_path=batch_prompt_path,
        narrative_col=narrative_col,
        label_col=label_col,
        narrative_label_col=narrative_label_col,
        batch_size=batch_size,
    )


def report(
    df: pd.DataFrame | None = None,
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    label_col: str = "label",
    narrative_label_col: str = "label_narrative",
    mismatch_csv: Path = DEFAULT_MISMATCH_CSV,
) -> None:
    if df is None:
        path = output_path if output_path.is_file() else input_path
        if not path.is_file():
            raise FileNotFoundError(f"No table found at {output_path} or {input_path}")
        df = _read_table(path)

    if narrative_label_col not in df.columns:
        print(f"Column '{narrative_label_col}' not found; run batch labeling first.")
        return

    filled = ~df[narrative_label_col].map(_is_empty_cell)
    if not filled.any():
        print("No label_narrative values yet.")
        return

    existing = _normalize_label_series(df[label_col])
    narrative = _normalize_label_series(df[narrative_label_col])
    df = df.copy()
    df["label_narrative_mismatch"] = existing != narrative

    subset = df[filled]
    n = len(subset)
    n_mis = int((existing[filled] != narrative[filled]).sum())
    print()
    print(f"Comparison `{label_col}` vs `{narrative_label_col}` (rows with narrative label):")
    print(f"  labeled rows: {n}")
    print(f"  mismatches: {n_mis} ({100 * n_mis / max(n, 1):.1f}%)")
    print()
    print("Mismatch matrix (existing label -> narrative label):")
    cross = pd.crosstab(existing[filled], narrative[filled], dropna=False)
    print(cross.to_string())

    for old, new in [
        ("benign", "malignant"),
        ("benign", "high risk"),
        ("malignant", "benign"),
        ("malignant", "high risk"),
        ("high risk", "malignant"),
        ("high risk", "benign"),
    ]:
        cnt = int(((existing == old) & (narrative == new) & filled).sum())
        if cnt:
            print(f"  {old} -> {new}: {cnt}")

    mismatch = df[df["label_narrative_mismatch"] & filled].copy()
    if len(mismatch):
        cols = [
            c for c in [
                "newaccession",
                "oldaccession",
                label_col,
                narrative_label_col,
                "label_narrative_rationale",
                "Narrative",
            ]
            if c in mismatch.columns
        ]
        mismatch_csv = Path(mismatch_csv)
        mismatch_csv.parent.mkdir(parents=True, exist_ok=True)
        mismatch[cols].to_csv(mismatch_csv, index=False)
        print(f"Mismatches written to {mismatch_csv} ({len(mismatch)} rows)")


def run_llm(
    input_path: Path,
    output_path: Path,
    *,
    narrative_col: str = "Narrative",
    label_col: str = "label",
    narrative_label_col: str = "label_narrative",
    rationale_col: str = "label_narrative_rationale",
    limit: Optional[int] = None,
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 6,
    skip_filled: bool = True,
    mismatch_csv: Path = DEFAULT_MISMATCH_CSV,
) -> None:
    from define_laterality import _make_llm, PROVIDER_DEFAULT_MODEL

    df = _ensure_label_columns(
        _load_working_df(input_path, output_path),
        narrative_label_col=narrative_label_col,
        rationale_col=rationale_col,
    )

    if skip_filled:
        indices = _eligible_indices(
            df, narrative_col=narrative_col, narrative_label_col=narrative_label_col
        )
    else:
        indices = [int(i) for i in df.index]
    if limit is not None:
        indices = indices[: int(limit)]

    print(f"Total rows: {len(df)} | LLM to process: {len(indices)}")
    if not indices:
        report(df, label_col=label_col, narrative_label_col=narrative_label_col, mismatch_csv=mismatch_csv)
        _write_table(df, output_path)
        return

    llm = _make_llm(provider, model_name, temperature, max_retries)
    print(f"Provider: {provider} | Model: {model_name or PROVIDER_DEFAULT_MODEL[provider]}")
    structured_llm = llm.with_structured_output(LabelOutput)

    n_ok = n_err = n_empty = 0
    for index in tqdm(indices, desc="LLM label"):
        narrative = str(df.at[index, narrative_col]).strip()
        if not narrative or narrative.lower() == "nan":
            df.at[index, narrative_label_col] = "benign"
            df.at[index, rationale_col] = "Empty narrative; defaulted to benign."
            n_empty += 1
            continue
        try:
            result: LabelOutput = structured_llm.invoke(PROMPT.format(narrative=narrative))
            df.at[index, narrative_label_col] = result.label
            df.at[index, rationale_col] = result.rationale
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            df.at[index, rationale_col] = f"LLM error: {e}"
            n_err += 1

    print(f"LLM done: ok={n_ok} empty_narrative={n_empty} errors={n_err}")
    report(df, label_col=label_col, narrative_label_col=narrative_label_col, mismatch_csv=mismatch_csv)
    _write_table(df, output_path)


def _pathology_text(row: pd.Series, cols: tuple[str, ...] = PATHOLOGY_MALIGNANT_COLS) -> str:
    parts = []
    for col in cols:
        if col in row.index and not _is_empty_cell(row[col]):
            parts.append(str(row[col]).strip())
    return " | ".join(parts)


def pathology_indicates_malignant(row: pd.Series) -> bool:
    """True when biopsy/pathology fields confirm malignancy (ICD-O behavior or classification)."""
    for col in PATHOLOGY_MALIGNANT_COLS:
        if col not in row.index or _is_empty_cell(row[col]):
            continue
        text = str(row[col]).strip().lower()
        if text in {"benign", "high risk"}:
            continue
        if "malignant" in text or "carcinoma" in text:
            return True
    return False


def pathology_behavior_is_in_situ_noninvasive(row: pd.Series) -> bool:
    """True when behaviorcodeicdo3description indicates in-situ / non-invasive carcinoma (DCIS-like)."""
    col = PATHOLOGY_BEHAVIOR_COL
    if col not in row.index or _is_empty_cell(row[col]):
        return False
    text = str(row[col]).strip().lower()
    return any(re.search(pat, text) for pat in PATHOLOGY_IN_SITU_PATTERNS)


def compute_v3_label(
    old_label: str,
    narrative_label: str,
    row: pd.Series,
) -> tuple[str, str]:
    """Apply label reconciliation rules for matches_birads4_all_v3."""
    old = _normalize_label(old_label)
    narr = _normalize_label(narrative_label)
    if old not in ALLOWED_LABELS:
        return old, "old_label invalid or empty; kept unchanged"

    if _is_empty_cell(narrative_label) or narr not in ALLOWED_LABELS:
        return old, "no narrative label; kept old_label"

    if narr == "high risk" and pathology_indicates_malignant(row):
        path = _pathology_text(row)
        if pathology_behavior_is_in_situ_noninvasive(row):
            return (
                "high risk",
                f"high risk narrative with in-situ/non-invasive pathology ({path}); categorized as high risk",
            )
        return (
            "malignant",
            f"high risk narrative with pathology confirms invasive malignancy ({path})",
        )

    if old == "high risk":
        return "high risk", "high risk rows kept unchanged"

    if old == "benign" and narr == "malignant":
        if pathology_indicates_malignant(row):
            path = _pathology_text(row)
            if pathology_behavior_is_in_situ_noninvasive(row):
                return (
                    "high risk",
                    f"benign->malignant narrative with in-situ/non-invasive pathology ({path}); categorized as high risk",
                )
            return "malignant", f"benign->malignant narrative with pathology confirms invasive malignancy ({path})"
        return "benign", "benign->malignant narrative but no confirming pathology; kept benign"

    if old == "malignant" and narr == "benign":
        if pathology_indicates_malignant(row):
            path = _pathology_text(row)
            return "malignant", f"malignant->benign narrative but pathology confirms malignancy ({path})"
        return "benign", "malignant->benign narrative; no confirming pathology; updated to benign"

    return old, "kept old_label (narrative mismatch not in reconciliation scope)"


def build_v3(
    input_path: Path,
    output_path: Path,
    *,
    narrative_label_col: str = "label_narrative",
    label_col: str = "label",
    old_label_col: str = "old_label",
    reason_col: str = "label_update_reason",
    narrative_labels_path: Path = DEFAULT_OUTPUT,
) -> None:
    """Build v3 table: old_label + reconciled label per narrative audit rules."""
    src_path = narrative_labels_path if narrative_labels_path.is_file() else Path(input_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"No source table at {narrative_labels_path} or {input_path}")
    df = _read_table(src_path)
    if narrative_label_col not in df.columns:
        raise ValueError(f"Missing {narrative_label_col}; run narrative labeling first.")

    if label_col not in df.columns:
        raise ValueError(f"Missing {label_col}")

    df = df.copy()
    df[old_label_col] = df[label_col]
    new_labels: list[str] = []
    reasons: list[str] = []
    for _, row in df.iterrows():
        label, reason = compute_v3_label(row[old_label_col], row[narrative_label_col], row)
        new_labels.append(label)
        reasons.append(reason)
    df[label_col] = new_labels
    df[reason_col] = reasons

    n_changed = int((df[old_label_col].map(_normalize_label) != df[label_col].map(_normalize_label)).sum())
    print(f"Source: {src_path}")
    print(f"Rows: {len(df)} | label changed from old_label: {n_changed}")
    print("New label counts:", df[label_col].value_counts().to_dict())
    changed = df[df[old_label_col].map(_normalize_label) != df[label_col].map(_normalize_label)]
    if len(changed):
        print("Change summary (old_label -> label):")
        cross = pd.crosstab(
            changed[old_label_col].map(_normalize_label),
            changed[label_col].map(_normalize_label),
        )
        print(cross.to_string())

    bm_candidates = df[
        (df[old_label_col].map(_normalize_label) == "benign")
        & (df[narrative_label_col].map(_normalize_label) == "malignant")
    ]
    if len(bm_candidates):
        bm_to_malignant = bm_candidates[bm_candidates[label_col].map(_normalize_label) == "malignant"]
        bm_to_high_risk = bm_candidates[bm_candidates[label_col].map(_normalize_label) == "high risk"]
        bm_kept = bm_candidates[bm_candidates[label_col].map(_normalize_label) == "benign"]
        bm_in_situ = bm_candidates[bm_candidates.apply(pathology_behavior_is_in_situ_noninvasive, axis=1)]
        print()
        print(f"Benign->malignant narrative candidates: {len(bm_candidates)}")
        print(f"  updated to malignant (invasive pathology): {len(bm_to_malignant)}")
        print(f"  updated to high risk (in-situ/non-invasive pathology): {len(bm_to_high_risk)}")
        print(f"  kept benign (no confirming pathology): {len(bm_kept)}")
        print(f"  in-situ/non-invasive behavior text among candidates: {len(bm_in_situ)}")
        if len(bm_to_malignant):
            all_confirmed = bool(bm_to_malignant.apply(pathology_indicates_malignant, axis=1).all())
            print(f"  all malignant rows have confirming pathology: {all_confirmed}")
            invasive_only = bm_to_malignant[
                ~bm_to_malignant.apply(pathology_behavior_is_in_situ_noninvasive, axis=1)
            ]
            print(f"  malignant rows without in-situ behavior text: {len(invasive_only)}")
            if not all_confirmed:
                bad = bm_to_malignant[~bm_to_malignant.apply(pathology_indicates_malignant, axis=1)]
                print(f"  rows malignant without pathology confirmation: {len(bad)}")

    hr_candidates = df[df[narrative_label_col].map(_normalize_label) == "high risk"]
    if len(hr_candidates):
        hr_path_malignant = hr_candidates[hr_candidates.apply(pathology_indicates_malignant, axis=1)]
        hr_in_situ = hr_candidates[hr_candidates.apply(pathology_behavior_is_in_situ_noninvasive, axis=1)]
        hr_to_malignant = hr_candidates[hr_candidates[label_col].map(_normalize_label) == "malignant"]
        hr_to_high_risk_path = hr_candidates[
            (hr_candidates[label_col].map(_normalize_label) == "high risk")
            & hr_candidates.apply(pathology_indicates_malignant, axis=1)
        ]
        print()
        print(f"High risk narrative rows: {len(hr_candidates)}")
        print(f"  with confirming pathology: {len(hr_path_malignant)}")
        print(f"  in-situ/non-invasive behavior text: {len(hr_in_situ)}")
        print(f"  label malignant (invasive pathology): {len(hr_to_malignant)}")
        print(f"  label high risk (in-situ pathology on HR narrative): {len(hr_to_high_risk_path)}")

    _write_table(df, output_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Narrative-based label audit for BI-RADS-4 table.")
    p.add_argument(
        "command",
        choices=[
            "prepare-batch",
            "merge-batch",
            "next-batch",
            "report",
            "run-llm",
            "run-all-batches",
            "build-v3",
        ],
        help="prepare-batch: export JSON for Cursor agent; merge-batch: import results; "
        "next-batch: merge-batch then prepare-batch; "
        "report: compare label vs label_narrative; run-llm: API labeling; "
        "run-all-batches: classify all remaining rows in-process; "
        "build-v3: write reconciled matches_birads4_all_v3.xlsx.",
    )
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--mismatch-csv", type=Path, default=DEFAULT_MISMATCH_CSV)
    p.add_argument("--narrative-col", default="Narrative")
    p.add_argument("--label-col", default="label")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--batch-json", type=Path, default=DEFAULT_BATCH_JSON)
    p.add_argument("--batch-results", type=Path, default=DEFAULT_BATCH_RESULTS_JSON)
    p.add_argument("--batch-prompt", type=Path, default=DEFAULT_BATCH_PROMPT)
    p.add_argument("--limit", type=int, default=None, help="run-llm: max rows.")
    p.add_argument("--no-skip-filled", action="store_true", help="run-llm: redo filled rows.")
    p.add_argument("--provider", default="openai", choices=["openai", "google"])
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_retries", type=int, default=6)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "prepare-batch":
        prepare_batch(
            args.input,
            args.output,
            batch_json=args.batch_json,
            batch_prompt_path=args.batch_prompt,
            narrative_col=args.narrative_col,
            label_col=args.label_col,
            batch_size=args.batch_size,
        )
    elif args.command == "merge-batch":
        merge_batch(
            args.input,
            args.output,
            batch_results_json=args.batch_results,
            label_col=args.label_col,
            mismatch_csv=args.mismatch_csv,
        )
    elif args.command == "next-batch":
        next_batch(
            args.input,
            args.output,
            batch_json=args.batch_json,
            batch_prompt_path=args.batch_prompt,
            batch_results_json=args.batch_results,
            narrative_col=args.narrative_col,
            label_col=args.label_col,
            batch_size=args.batch_size,
            mismatch_csv=args.mismatch_csv,
        )
    elif args.command == "report":
        report(
            input_path=args.input,
            output_path=args.output,
            label_col=args.label_col,
            mismatch_csv=args.mismatch_csv,
        )
    elif args.command == "run-llm":
        run_llm(
            args.input,
            args.output,
            narrative_col=args.narrative_col,
            label_col=args.label_col,
            limit=args.limit,
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            skip_filled=not args.no_skip_filled,
            mismatch_csv=args.mismatch_csv,
        )
    elif args.command == "run-all-batches":
        run_all_batches(
            args.input,
            args.output,
            narrative_col=args.narrative_col,
            label_col=args.label_col,
            batch_size=args.batch_size,
            mismatch_csv=args.mismatch_csv,
        )
    elif args.command == "build-v3":
        v3_out = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_V3_OUTPUT
        build_v3(args.input, v3_out, label_col=args.label_col)
