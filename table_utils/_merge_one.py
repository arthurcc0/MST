import json
from pathlib import Path

import pandas as pd

p = Path(__file__).resolve().parent
df = pd.read_csv(p / "lat_added_dummy_ehr_chat.csv")
with open(p / "_batch_results.json", encoding="utf-8-sig") as f:
    results = json.load(f)
for x in results:
    i = int(x["idx"])
    df.at[i, "laterality"] = float(x["code"])
    df.at[i, "lateralityDescription"] = x["description"]
df.to_csv(p / "lat_added_dummy_ehr_chat.csv", index=False)

lat = pd.to_numeric(df["laterality"], errors="coerce")
desc_na = df["lateralityDescription"].isna() | (
    df["lateralityDescription"].astype(str).str.strip() == ""
)
narrative = df["Narrative"].fillna("").astype(str).str.strip()
eligible = lat.isna() & desc_na & (narrative != "") & (narrative.str.lower() != "nan")
next_idx = df.index[eligible].tolist()[:25]
next_batch = [{"idx": int(i), "narrative": df.at[i, "Narrative"]} for i in next_idx]
with open(p / "_batch.json", "w", encoding="utf-8") as f:
    json.dump(next_batch, f, indent=1)
print(f"merged {len(results)} rows | eligible_remaining {int(eligible.sum())} | next_batch {len(next_batch)}")
if next_batch:
    print(f"idx_range {next_batch[0]['idx']} .. {next_batch[-1]['idx']}")
