from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default=r'\\rad-maid-004\D\Duke-Cancer_MRI')
parser.add_argument('--path_preprocessed', type=str, default=r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed_crop_n4bc_plhe_fast_full-a')
args = parser.parse_args()

path_root_in = Path(args.path_data)
path_root_out = Path(args.path_preprocessed)

# Load clinical data
df_clinical = pd.read_excel(path_root_in/'Clinical_and_Other_Features.xlsx', header=[0, 1, 2])
df_clinical = df_clinical[df_clinical[df_clinical.columns[38]] != 'NC']
df_clinical = df_clinical[[df_clinical.columns[0], df_clinical.columns[36], df_clinical.columns[38]]]
df_clinical.columns = ['PatientID', 'Location', 'Bilateral']
df_clinical['PatientID'] = df_clinical['PatientID'].str.split('_').str[2].astype(int)

# Load file mapping to check for segmentations
df_mapping = pd.read_csv(path_root_in/'Radiology_and_Pathology_Features.csv')
segmentation_patients = df_mapping[df_mapping['SequenceName'] == 'Segmentation']['PatientID'].unique()

dfs = []
for side in ["left", 'right']:
    df_side = pd.DataFrame({
        'PatientID': df_clinical["PatientID"],
        'UID': df_clinical["PatientID"].astype(str).str.zfill(3) + f"_{side}",
        'Malignant': df_clinical[["Location", "Bilateral"]].apply(lambda ds: int((ds[0] == side[0].upper()) | (ds[1] == 1)), axis=1)
    })
    dfs.append(df_side)

df = pd.concat(dfs, ignore_index=True)

# Add has_segmentation flag
df['has_segmentation'] = df['PatientID'].isin(segmentation_patients)

df = df.reset_index(drop=True)
splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['UID'], df['Malignant'], groups=df['PatientID'])):
    df_split = df.copy()
    df_split['Fold'] = fold_i 
    df_trainval = df_split.loc[train_val_idx]
    train_idx, val_idx = list(sgkf2.split(df_trainval['UID'], df_trainval['Malignant'], groups=df_trainval['PatientID']))[0]
    train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
    df_split.loc[train_idx, 'Split'] = 'train' 
    df_split.loc[val_idx, 'Split'] = 'val' 
    df_split.loc[test_idx, 'Split'] = 'test' 
    splits.append(df_split)
df_splits = pd.concat(splits)

path_out = path_root_out/'splits'
path_out.mkdir(parents=True, exist_ok=True)
df_splits.to_csv(path_out/'split.csv', index=False)