from pathlib import Path
import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

base_dir = Path(r'D:\PENN-MRI')

path_root_in = base_dir
path_root_out = base_dir / 'penn-preprocessed_cropped2'
path_root_out.mkdir(parents=True, exist_ok=True)

clinical_file = 'penn-preprocessed_mapping.csv'
df_clinical = pd.read_csv(clinical_file, dtype={'PatientID': str})

laterality_file = 'label_added_dummy_ehr_new.csv'
df_laterality = pd.read_csv(laterality_file, dtype={'dummy_acc': str})

malignancy_file = Path(r'D:\Users\UFPB\gabriel ayres\MST\table_utils\new_acc_new.csv') 
df_malignancy_determine = pd.read_csv(malignancy_file, dtype=str)

exclude_file = Path(r'table_utils\bc_after_exclude.csv')
print(f"Loading exclude file: {exclude_file}")
print(f"Number of patients : {len(df_clinical)}")

columns_to_read = ['birads2after_acc', 'birads45after_acc', 'birads6after_acc']
df_exclude = pd.read_csv(exclude_file, usecols=columns_to_read, dtype=str)

# Strip whitespace from each column before processing
for col in columns_to_read:
    df_exclude[col] = df_exclude[col].str.strip()

exclude_1 = df_exclude['birads2after_acc'].dropna().unique()
exclude_2 = df_exclude['birads45after_acc'].dropna().unique()
exclude_3 = df_exclude['birads6after_acc'].dropna().unique()
exclude = np.concatenate((exclude_1, exclude_2, exclude_3))
exclude = exclude.tolist()

df_clinical['dummy_acc'] = df_clinical['PatientID'].apply(lambda x: str(x).split('_')[0].strip())

print("--- Debugging Exclusion ---")
print(f"Total IDs in exclusion list: {len(exclude)}")
print(f"First 5 'exclude' values: {exclude[:5]}")
print(f"First 5 'dummy_acc' values: {df_clinical['dummy_acc'].head().tolist()}")

initial_patient_count = len(df_clinical)
df_clinical = df_clinical[~df_clinical['dummy_acc'].isin(exclude)]
final_patient_count = len(df_clinical)

print(f"Excluded {initial_patient_count - final_patient_count} patients")
print(f"Remaining patients: {final_patient_count}")
df_merged = pd.merge(df_clinical, df_laterality[['dummy_acc', 'laterality']], on='dummy_acc', how='left')
patient_ids = df_merged['dummy_acc'].unique()

dfs = []
for side in ["left", 'right']:
    df_side = pd.DataFrame({
        'dummy_acc': patient_ids,
        'UID': [f"{pid}_{side}" for pid in patient_ids],
        'side': side
    })
    dfs.append(df_side)
df_expanded = pd.concat(dfs, ignore_index=True)

df_final = pd.merge(df_expanded, df_laterality[['dummy_acc', 'laterality', 'label']], on='dummy_acc', how='left')

def process_benign_laterality(row):
    # handle laterality for benign and birads 4 benign confirmed by biopsycases
    if pd.isna(row['laterality']) and is_benign_case(row):
        return np.random.randint(1, 3)  # Returns 1 or 2
    return row['laterality']

def is_benign_case(row):
    if 'label' in row.index:
        return row['label'] in df_malignancy_determine['birads2noprior_acc'].unique()
    return False

def assign_side_label(row):
    # First, determine if the patient has any malignancy
    is_malignant_patient = (row['dummy_acc'] in df_malignancy_determine['birads45noprior_acc'].unique()) or \
                           (row['dummy_acc'] in df_malignancy_determine['birads6noprior_acc'].unique()) or \
                           (row['dummy_acc'] in df_malignancy_determine['birads6prior_acc'].unique())

    if not is_malignant_patient:
        return 0 # Benign case, both sides are benign

    # Malignant case, check laterality
    # Assuming 'laterality' column uses 1 for Right and 2 for Left
    side_map = {'right': 1, 'left': 2}
    
    if pd.isna(row['laterality']):
        return 0 # Or handle as an error/default case if laterality is missing for a malignant case

    if side_map.get(row['side']) == row['laterality']:
        return 1 # This is the malignant side
    else:
        return 0 # This is the contralateral benign side

df_final['Malignant'] = df_final.apply(assign_side_label, axis=1)
df_final['laterality'] = df_final.apply(process_benign_laterality, axis=1)

df = df_final.copy()
splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['UID'], df['Malignant'], groups=df['dummy_acc'])):
    df_split = df.copy()
    df_split['Fold'] = fold_i
    df_trainval = df_split.loc[train_val_idx]

    if len(df_trainval) > 1 and len(set(df_trainval['Malignant'])) > 1:
        train_idx_local, val_idx_local = next(sgkf2.split(df_trainval['UID'], df_trainval['Malignant'], groups=df_trainval['dummy_acc']))
        train_idx = df_trainval.iloc[train_idx_local].index
        val_idx = df_trainval.iloc[val_idx_local].index

        df_split.loc[train_idx, 'Split'] = 'test'
        df_split.loc[val_idx, 'Split'] = 'test'
    else: 
        df_split.loc[train_val_idx, 'Split'] = 'test'

    df_split.loc[test_idx, 'Split'] = 'test'
    splits.append(df_split)

df_splits = pd.concat(splits)

df_splits.drop_duplicates(subset=['UID'], keep='first', inplace=True)
df_splits.dropna(subset=['label'], inplace=True)
#output_file = path_root_out / 'penn-preprocessed_datasplit.csv'
output_file = 'penn-preprocessed_datasplit.csv'
df_splits.to_csv(output_file, index=False)
print(f"Data splits saved to {output_file}")


