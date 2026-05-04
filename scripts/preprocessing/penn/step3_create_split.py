from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Define the base directory for the Penn dataset
base_dir = Path(r'\\rad-maid-004\D\PENN-MRI')

# Define input and output paths
path_root_in = base_dir
path_root_out = base_dir / 'preprocessed_cropped'
path_root_out.mkdir(parents=True, exist_ok=True)

# Load clinical data and laterality info
clinical_file = 'penn_mapping.csv'
df_clinical = pd.read_csv(clinical_file, dtype={'PatientID': str})

laterality_file = 'mapped_patient_info.csv'
df_laterality = pd.read_csv(laterality_file, dtype={'dummy_acc': str})

# Prepare dataframes for merging
df_clinical['dummy_acc'] = df_clinical['PatientID'].apply(lambda x: x.split('_')[0])
df_merged = pd.merge(df_clinical, df_laterality[['dummy_acc', 'laterality']], on='dummy_acc', how='left')

# Get unique patient accession numbers
patient_ids = df_merged['dummy_acc'].unique()

# Create a full list of all possible cases (left and right for each patient)
dfs = []
for side in ["left", 'right']:
    df_side = pd.DataFrame({
        'dummy_acc': patient_ids,
        'UID': [f"{pid}_{side}" for pid in patient_ids],
        'side': side
    })
    dfs.append(df_side)
df_expanded = pd.concat(dfs, ignore_index=True)

# Merge with laterality info to determine malignancy
df_final = pd.merge(df_expanded, df_laterality[['dummy_acc', 'laterality']], on='dummy_acc', how='left')

# Determine Malignancy: 1.0 for Right, 2.0 for Left
def is_malignant(row):
    if row['side'] == 'right' and row['laterality'] == 1.0:
        return 1
    if row['side'] == 'left' and row['laterality'] == 2.0:
        return 1
    return 0

df_final['Malignant'] = df_final.apply(is_malignant, axis=1)

# Prepare for splitting
df = df_final.copy()
splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['UID'], df['Malignant'], groups=df['dummy_acc'])):
    df_split = df.copy()
    df_split['Fold'] = fold_i
    df_trainval = df_split.loc[train_val_idx]

    # Check if there are enough samples and classes for a further split
    if len(df_trainval) > 1 and len(set(df_trainval['Malignant'])) > 1:
        train_idx_local, val_idx_local = next(sgkf2.split(df_trainval['UID'], df_trainval['Malignant'], groups=df_trainval['dummy_acc']))
        train_idx = df_trainval.iloc[train_idx_local].index
        val_idx = df_trainval.iloc[val_idx_local].index

        df_split.loc[train_idx, 'Split'] = 'test'
        df_split.loc[val_idx, 'Split'] = 'test'
    else:  # Handle cases where train-val set is too small or has only one class
        df_split.loc[train_val_idx, 'Split'] = 'test'

    df_split.loc[test_idx, 'Split'] = 'test'
    splits.append(df_split)

df_splits = pd.concat(splits)

# Save the splits to a CSV file
output_file = path_root_out / 'datasplit.csv'
df_splits.to_csv(output_file, index=False)
print(f"Data splits saved to {output_file}")


