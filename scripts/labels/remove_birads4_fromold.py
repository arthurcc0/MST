"""Drop accessions that appear in both old and new Penn tables from the old table."""

import pandas as pd
from pathlib import Path

OLD_TABLE = Path(r"D:\Users\arthur\Projects\MST\table_utils\lat_added_dummy_ehr_chat.csv")
NEW_TABLE = Path(r"D:\Users\arthur\Projects\MST\tables\matches_birads4_all.xlsx")
OUTPUT = Path(r"D:\Users\arthur\Projects\MST\table_utils\lat_added_dummy_ehr_chat_no_birads4.csv")

OLD_ACC = "dummy_acc"
NEW_ACC = "newaccession"

old_df = pd.read_csv(OLD_TABLE, dtype=str)
new_df = pd.read_excel(NEW_TABLE, dtype=str)

old_df[OLD_ACC] = old_df[OLD_ACC].str.strip()
new_accs = set(new_df[NEW_ACC].str.strip().dropna())

overlap = old_df[OLD_ACC].isin(new_accs)
print(f"Old rows: {len(old_df)}")
print(f"New accessions: {len(new_accs)}")
print(f"Removing {overlap.sum()} rows present in both tables")

old_df = old_df.loc[~overlap].reset_index(drop=True)
old_df.to_csv(OUTPUT, index=False)
print(f"Saved {len(old_df)} rows to {OUTPUT}")
