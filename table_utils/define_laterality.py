"""Fill the ``laterality`` column of an EHR-style CSV using LLM extraction.

Rules:

* Rows where the study-level assessment indicates BI-RADS 1 / Negative are
  treated as "no laterality applies": ``laterality`` is left blank and
  ``lateralityDescription`` is filled with a short note (no LLM call).
* For every other row that still has an empty ``laterality``, the Narrative
  is sent to an LLM that returns a structured (code, description) pair via
  ``with_structured_output``.

Supported providers (selectable via ``--provider``):

* ``openai`` (default) — uses ``langchain-openai`` and ``OPENAI_API_KEY``.
  Default model: ``gpt-4o``.
* ``google`` — uses ``langchain-google-genai`` and ``GOOGLE_API_KEY`` (or
  ``GEMINI_API_KEY``). Default model: ``gemini-2.0-flash``. The free tier
  is rate-limited (~15 RPM); ``--max_retries`` provides exponential backoff
  on 429 errors so a long run will throttle itself rather than fail.

Codes:

    0 = UNSPECIFIED   1 = RIGHT   2 = LEFT   3 = BILATERAL   4 = IMPLANTS
    5 = POST‑MASTECTOMY CONTEXT (mastectomy/reconstruction dominates)

Code **4**: implants override lesion laterality except when an implant is
only on one side and the dominant finding is in the contralateral breast.

Code **5** (**from manual / LLM rules**): use when **mastectomy has removed
(or replaced with flap/reconstruction)** one or **both** breasts in a way
that dominates the lateral assignment:

- **Bilateral mastectomy** (or bilateral post‑mastectomy reconstruction as
  the main frame of the study) → **5**.
- **Unilateral mastectomy** (single‑sided mastectomy/flap/reconstruction):

  - If **all dominant localized imaging findings** (mass, NME, recurrence
    workup targets, operative clips on the lesion path, etc.) are **only**
    on the **contralateral breast** (the remaining native breast) → assign
    **regular codes 0–4** according to those findings (typically **1** or **2**
    plus **3**/**4**/**0** as usual).
  - **Otherwise** (findings anchored to the **mastectomy**/reconstructed side,
    bilateral post‑operative fields, ambiguous split, etc.) → **5**.

Usage::

    python define_laterality.py --input ehr.csv --output ehr_with_lat.csv
    python define_laterality.py -i ehr.csv -o out.csv --limit 10        # test run
    python define_laterality.py -i ehr.csv -o out.csv --provider google
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm


load_dotenv()


# Default chat model per provider (overridable via --model).
PROVIDER_DEFAULT_MODEL = {
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash",
}


def _make_llm(provider: str, model_name: Optional[str], temperature: float, max_retries: int):
    """Construct a chat LLM for the chosen provider.

    Provider packages are imported lazily so a Gemini run doesn't require
    ``langchain-openai`` to be installed (and vice versa).
    """
    provider = provider.lower().strip()
    if provider not in PROVIDER_DEFAULT_MODEL:
        raise ValueError(f"Unknown provider: {provider!r}. Choose one of {list(PROVIDER_DEFAULT_MODEL)}.")
    model = model_name or PROVIDER_DEFAULT_MODEL[provider]

    if provider == "openai":
        try:
            from langchain_openai.chat_models import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in a .env file at the project root, then rerun."
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_retries=max_retries,
        )

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "langchain-google-genai is not installed. Run: "
                "pip install langchain-google-genai"
            ) from e
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Get a free key from "
                "https://aistudio.google.com/app/apikey and put it in your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            max_retries=max_retries,
        )

    # Unreachable: validated above.
    raise AssertionError(provider)


PROMPT = """\
Extract the laterality (left, right, or bilateral) of breast lesions/tumors from the following radiology narrative report.

TASK:
Analyze the provided radiology report and identify all mentions of breast lesions, masses, tumors, or other localized findings (e.g., biopsy clips, artifacts, architectural distortion). For each finding, determine its laterality based on explicit anatomical references.

LATERALITY CLASSIFICATION CODES:
0 = UNSPECIFIED: Lesions mentioned without clear laterality indication,
    OR the only located observation is a post-procedural change (biopsy
    clip, susceptibility artifact, postsurgical scar) on an otherwise
    negative report.
1 = RIGHT: Lesional findings specifically in the right breast
2 = LEFT: Lesional findings specifically in the left breast
3 = BILATERAL: Lesional findings present in both breasts (bilateral
    simple cysts count; bilateral background parenchymal enhancement
    does not)
4 = IMPLANTS: Breast implant(s) are present. **Implants take priority over
    separate lesional findings (masses, NME, cysts, etc.) — use code 4 when
    implants are documented, unless the report clearly states an implant
    **only in one breast** and the main finding is in the **opposite**
    breast (then code 1/2/3 for that finding and mention the single-sided
    implant in the description). Bilateral implants with any parenchymal
    finding still → 4.
5 = POST‑MASTECTOMY CONTEXT: **Mastectomy / reconstruction breast(s) dominate**
    so native left‑vs‑right lesion tagging does not apply in the usual way.

    • **Bilateral mastectomy** (or bilateral reconstructed breasts as the main
      clinical/imaging framework, e.g. bilateral TRAM/flap reconstruction) → **5**.

    • **Unilateral mastectomy only** (one breast removed or reconstructed):

      – If **all dominant localized findings** (recurrence‑suspected mass/NME,
        actionable clips, asymmetric tumor bed workup **as the lesion target**) lie
        **only** in the **contralateral (remaining native) breast** → use **0–4**
        for that breast’s findings (**not** 5).

      – Otherwise (targets on mastectomy/flap side, bilateral operative fields both
        described, recurrence surveillance explicitly on reconstructed side **as the
        index**, unclear split) → **5**.

    If both **5** (mastectomy framing) and **4** (implants) appear, prefer **5**
    when bilateral mastectomy/reconstruction narrative is dominant; otherwise follow
    the unilateral mastectomy branch above vs. standard implant rules.

KEY INDICATORS TO LOOK FOR:

Right Breast Indicators (Code: 1):
- "right breast"
- "right breast artifact"
- "right axillary"
- "right upper/lower/outer/inner quadrant"
- "right retroareolar"
- "right subareolar"
- Anatomical descriptions following "Right:" section headers

Left Breast Indicators (Code: 2):
- "left breast"
- "left breast artifact"
- "left axillary"
- "left upper/lower/outer/inner quadrant"
- "left retroareolar"
- "left subareolar"
- Anatomical descriptions following "Left:" section headers

Bilateral Indicators (Code: 3):
- "bilateral" or "bilaterally" when describing a finding, not the procedure.
- "both breasts" when describing findings.
- Lesions explicitly described in both the left and right breast sections.

ANALYSIS GUIDELINES:

Context Clues:
- Pay attention to section headers (e.g., "Left:", "Right:")
- Consider anatomical quadrant descriptions
- Note axillary lymph node involvement
- Look for comparative language ("compared to the contralateral breast")

Common Pitfalls to Avoid:
- Don't assume laterality from previous mentions without explicit confirmation
- Be careful with pronouns - ensure they refer to the correct anatomical side
- Watch for lesions described in relation to biopsy clips or markers
- Consider that multiple lesions may have different lateralities
- **Implants vs. lesions:** If bilateral (or any contralateral pair of) implants are present, prefer code 4 over 1/2/3 unless the text clearly places implant(s) in only one breast and the dominant actionable finding is only in the opposite breast.
- **Mastectomy vs. lesion codes:** Apply code **5** per the definition above **before** defaulting to 1/2/3 for post‑mastectomy reconstructed breasts. Do **not** use 5 for simple lumpectomy or segmental partial surgery without mastectomy framing.
- **Distinguish Procedure vs. Finding:** The imaging technique (e.g., 'Bilateral MRI') may be bilateral, but this does not mean the findings are bilateral. Base your laterality code ONLY on the location of the actual lesions, masses, or suspicious findings described in the report.
- **Include All Findings:** Do not ignore a finding just because it is described as 'benign,' 'stable,' or is a post-procedural change (e.g., 'biopsy artifact'). If a finding has a specific location, its laterality must be coded.

Ambiguous Cases:
- If laterality cannot be definitively determined, classify as UNSPECIFIED (Code: 0)
- Note any contextual information that might suggest laterality
- Distinguish between primary tumors and metastatic involvement

Now analyze the following radiology report and return a structured response with:
- laterality_code: the single-digit code (0, 1, 2, 3, 4, or 5)
- laterality_description: a brief (1-2 sentence) explanation citing the report phrases that drove that code.

Radiology Report:
{radiology_report}
"""


class LateralityOutput(BaseModel):
    """Structured laterality extraction result."""

    laterality_code: Literal[0, 1, 2, 3, 4, 5] = Field(
        description=(
            "0=UNSPECIFIED, 1=RIGHT, 2=LEFT, 3=BILATERAL, 4=IMPLANTS, "
            "5=POST‑MASTECTOMY CONTEXT (bilateral mastectomy; or unilateral "
            "mastectomy with findings NOT confined to contralateral native breast). "
            "Use 4 when implants dominate per prompt; use 5 when mastectomy "
            "framing dominates per prompt (5 may supersede 4 for bilateral mastectomy)."
        ),
    )
    laterality_description: str = Field(
        description="Short (1-2 sentence) explanation that cites the phrases driving the code.",
    )


# Match BI-RADS 1 / "Negative" assessments. Conservative: accept variants like
# "1", "1:", "1: Negative", "BI-RADS 1", "BI RADS 1: Negative".
_NEG_LEADING_1 = re.compile(r"^\s*(?:BI[\s\-]?RADS\s*)?1\b", re.IGNORECASE)


def is_negative_assessment(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    if not s:
        return False
    if _NEG_LEADING_1.match(s):
        return True
    return "negative" in s.lower()


def define_laterality(
    input_csv: Path,
    output_csv: Path,
    narrative_col: str = "Narrative",
    assessment_col: str = "studylevelassessment",
    laterality_col: str = "laterality",
    description_col: str = "lateralityDescription",
    limit: Optional[int] = None,
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> None:
    df = pd.read_csv(input_csv)

    if narrative_col not in df.columns:
        raise KeyError(f"Narrative column '{narrative_col}' not found. Columns: {list(df.columns)}")
    if laterality_col not in df.columns:
        df[laterality_col] = pd.NA
    if description_col not in df.columns:
        df[description_col] = pd.NA

    df[narrative_col] = df[narrative_col].astype(str)

    # Rows that still need a value in `laterality`.
    needs_fill = df[laterality_col].isna()

    # ---- BI-RADS 1 / Negative branch (no LLM) -------------------------------
    n_negative_skipped = 0
    if assessment_col in df.columns:
        is_neg = df[assessment_col].apply(is_negative_assessment)
        neg_mask = needs_fill & is_neg
        if neg_mask.any():
            df.loc[neg_mask, description_col] = df.loc[neg_mask, assessment_col].apply(
                lambda v: f"Negative study (assessment='{v}'); no laterality applies."
            )
            n_negative_skipped = int(neg_mask.sum())
        # Done with these rows; don't pass them to the LLM.
        needs_fill = needs_fill & ~is_neg
    else:
        print(f"[warn] assessment column '{assessment_col}' not found; not skipping any Negative rows.")

    # ---- LLM branch ---------------------------------------------------------
    idx_to_llm = df.index[needs_fill].tolist()
    if limit is not None:
        idx_to_llm = idx_to_llm[: int(limit)]

    print(
        f"Total rows: {len(df)} | already filled: {int((~df[laterality_col].isna()).sum())} "
        f"| skipped (negative): {n_negative_skipped} | LLM to-do: {len(idx_to_llm)}"
    )

    if not idx_to_llm:
        _save(df, output_csv)
        return

    try:
        llm = _make_llm(provider, model_name, temperature, max_retries)
    except RuntimeError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        _save(df, output_csv)
        sys.exit(2)
    print(f"Provider: {provider} | Model: {model_name or PROVIDER_DEFAULT_MODEL[provider]} "
          f"| max_retries: {max_retries}")
    structured_llm = llm.with_structured_output(LateralityOutput)

    n_ok = n_err = n_empty = 0
    for index in tqdm(idx_to_llm, desc="LLM laterality"):
        narrative = df.at[index, narrative_col]
        if not isinstance(narrative, str) or not narrative.strip() or narrative.lower() == "nan":
            df.at[index, description_col] = "No narrative text; could not determine laterality."
            n_empty += 1
            continue
        try:
            result: LateralityOutput = structured_llm.invoke(
                PROMPT.format(radiology_report=narrative)
            )
            df.at[index, laterality_col] = float(result.laterality_code)
            df.at[index, description_col] = result.laterality_description
            n_ok += 1
        except Exception as e:  # noqa: BLE001 -- log and move on
            df.at[index, description_col] = f"LLM error: {e}"
            n_err += 1

    print(
        f"\nDone. filled_by_llm={n_ok}  empty_narrative={n_empty}  errors={n_err}  "
        f"negative_skipped={n_negative_skipped}"
    )
    _save(df, output_csv)


def _save(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fill the `laterality` column of an EHR CSV using LLM extraction from the Narrative."
    )
    p.add_argument("-i", "--input", required=True, type=Path, help="Input CSV path.")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output CSV path.")
    p.add_argument("--narrative_col", default="Narrative",
                   help="Name of the column holding the radiology narrative (default: Narrative).")
    p.add_argument("--assessment_col", default="studylevelassessment",
                   help=("Name of the column holding the study-level assessment. Rows whose value "
                         "matches BI-RADS 1 / Negative are skipped (no LLM call). Expected values "
                         "follow '<digit>: <description>' (e.g. '1: Negative', '4: Suspicious')."))
    p.add_argument("--laterality_col", default="laterality",
                   help="Name of the column to fill (default: laterality).")
    p.add_argument("--description_col", default="lateralityDescription",
                   help="Name of the description column to create/fill (default: lateralityDescription).")
    p.add_argument("--limit", type=int, default=None,
                   help="If given, process at most this many LLM rows (useful for test runs).")
    p.add_argument("--provider", default="openai",
                   choices=sorted(PROVIDER_DEFAULT_MODEL.keys()),
                   help="LLM provider (default: openai). 'google' uses Gemini via "
                        "langchain-google-genai and GOOGLE_API_KEY / GEMINI_API_KEY.")
    p.add_argument("--model", default=None,
                   help="Chat model name. Default depends on --provider: 'gpt-4o' for openai, "
                        "'gemini-2.0-flash' for google.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_retries", type=int, default=6,
                   help="Per-call retry budget. Provides exponential backoff on rate-limit (429) "
                        "errors; useful on Gemini's free 15-RPM tier (default: 6).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    define_laterality(
        input_csv=args.input,
        output_csv=args.output,
        narrative_col=args.narrative_col,
        assessment_col=args.assessment_col,
        laterality_col=args.laterality_col,
        description_col=args.description_col,
        limit=args.limit,
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
