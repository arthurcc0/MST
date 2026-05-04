

import logging
import sys
import traceback
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import functools

import numpy as np
import pandas as pd
import torch
import torchio as tio
from tqdm import tqdm

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Device Setup ---
device = torch.device("cuda")
logger.info(f"Using device: {device}")
# --------------------

def save_with_timeout(img, path, timeout=60):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(img.save, path)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise Exception(f"Save operation timed out after {timeout} seconds")

def npy2nifti(file_info, path_root_out_data):
    path_file = file_info['path_file']
    patient_id = file_info['patient_id']
    contrast = file_info['contrast'].lower()

    try:
        data = np.load(path_file)
        tensor = torch.from_numpy(data).to(device)
        rotated_tensor = torch.rot90(tensor, k=1, dims=(0, 1)).contiguous()
        tensor_4d = rotated_tensor.unsqueeze(0)
        img = tio.ScalarImage(tensor=tensor_4d.cpu())
        
        path_out_dir = path_root_out_data / patient_id
        path_out_dir.mkdir(parents=True, exist_ok=True)

        if 'pre' in contrast:
            filename = 'pre.nii.gz'
        elif 'post' in contrast:
            filename = 'post.nii.gz'
        else:
            filename = f"{contrast}.nii.gz"

        output_path = path_out_dir / filename
        try:
            logger.info(f"Saving file to {output_path}")
            save_with_timeout(img, output_path)
            logger.info(f"Successfully saved {output_path}")
        except Exception:
            logger.error(f"Failed during save for {path_file}:\n{traceback.format_exc()}")

    except Exception:
        logger.error(f"Failed to process {path_file}:\n{traceback.format_exc()}")


if __name__ == '__main__':
    path_root = Path(r'D:\PENN-MRI') 
    data_root_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI')

    path_root_in = data_root_dir / 'output2' / 'preproc' / 'n4bc_plhe'
    path_root_out = path_root/'penn-preprocessed2'
    path_root_out_data = path_root_out/'data'
    


    path_root_out_data.mkdir(parents=True, exist_ok=True)
    df_patient_info = pd.read_csv(Path(r'D:\PENN-MRI\label_added_dummy_ehr_new.csv'), dtype={'dummy_acc': str})
    valid_patients = df_patient_info['dummy_acc'].unique()

    df_mapping = pd.read_csv(Path(r'D:\PENN-MRI\penn-preprocessed_mapping.csv'), dtype={'PatientID': str})
    df_mapping = df_mapping[df_mapping['PatientID'].isin(valid_patients)]

    files_to_process = []
    for _, row in df_mapping.iterrows():
        path_file = Path(row['FullPath'])
        if path_file.exists():
            files_to_process.append({
                'path_file': path_file,
                'patient_id': row['PatientID'],
                'contrast': row['Contrast']
            })
    
    logger.info(f"Found {len(files_to_process)} .npy files to process after filtering.")

    processor = functools.partial(npy2nifti,
                                path_root_out_data=path_root_out_data)

    # Process files
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(processor, files_to_process), total=len(files_to_process)):
            pass

    print("\nConversion process complete.")
