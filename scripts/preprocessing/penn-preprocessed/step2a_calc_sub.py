from pathlib import Path
import logging
import SimpleITK as sitk
import numpy as np
import torch
from multiprocessing import Pool
from tqdm import tqdm
import os
logger = logging.getLogger(__name__)
device = torch.device("cuda")

def process(path_patient):
    temp_output_path = None
    patient_id = path_patient.name
    try:
        logger.debug(f"Processing {patient_id}")
        path_pre = path_patient / 'pre.nii.gz'
        path_post = path_patient / 'post.nii.gz'

        if not path_pre.exists() or not path_post.exists():
            logger.warning(f"Skipping {patient_id}: missing pre or post contrast images.")
            return

        pre_nii = sitk.ReadImage(os.path.normpath(str(path_pre)), sitk.sitkInt32)
        post_nii_orig = sitk.ReadImage(os.path.normpath(str(path_post)), sitk.sitkInt32)

        # logger.debug(f"Resampling post.nii.gz to match pre.nii.gz space for {patient_id}")
        post_nii = sitk.Resample(post_nii_orig, pre_nii, sitk.Transform(), sitk.sitkLinear, 0, pre_nii.GetPixelID())

        pre_arr = sitk.GetArrayFromImage(pre_nii)
        post_arr = sitk.GetArrayFromImage(post_nii)

        # pre_tensor = torch.from_numpy(pre_arr).to(device)
        # post_tensor = torch.from_numpy(post_arr).to(device)

        sub_arr = post_arr - pre_arr
        sub_arr = sub_arr - sub_arr.min()
        sub_arr = sub_arr.astype(np.uint16)

        sub_nii = sitk.GetImageFromArray(sub_arr)
        sub_nii.CopyInformation(pre_nii)

        output_path = path_patient / 'sub.nii.gz'

        try:
            sitk.WriteImage(sub_nii, str(output_path))
            logger.debug(f"Saved subtraction image to {output_path}")
        except Exception as write_error:
            logger.error(f"Failed to write file for {patient_id}: {write_error}")
    except Exception as e:
        logger.error(f"Failed to process {patient_id} due to error: {e}")



if __name__ == "__main__":

    path_root = Path(r'D:\PENN-MRI') 
    path_root_out = path_root / 'penn-preprocessed2'
    path_root_out_data = path_root_out / 'data'

    patient_dirs = [p for p in path_root_out_data.iterdir() if p.is_dir()]
    logger.info(f"Found {len(patient_dirs)} patient directories to process.")
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process, patient_dirs), total=len(patient_dirs)):
            pass

    logger.info("Subtraction calculation complete.")

        
    