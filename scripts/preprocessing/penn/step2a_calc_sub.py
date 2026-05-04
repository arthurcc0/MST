from pathlib import Path 
import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm

logger = logging.getLogger(__name__)

def process(path_patient):
    patient_id = path_patient.name
    path_pre = path_patient / 'pre.nii.gz'
    path_post = path_patient / 'post.nii.gz'

    if not path_pre.exists() or not path_post.exists():
        logger.warning(f"Skipping {patient_id}: missing pre or post contrast images.")
        return

    logger.debug(f"Processing {patient_id}")
    pre_nii = sitk.ReadImage(str(path_pre), sitk.sitkInt32)
    post_nii_orig = sitk.ReadImage(str(path_post), sitk.sitkInt32)
    
    # logger.debug(f"Resampling post.nii.gz to match pre.nii.gz space for {patient_id}")
    post_nii = sitk.Resample(post_nii_orig, pre_nii, sitk.Transform(), sitk.sitkLinear, 0, pre_nii.GetPixelID())

    pre_arr = sitk.GetArrayFromImage(pre_nii)
    post_arr = sitk.GetArrayFromImage(post_nii) 
    
    sub_arr = post_arr - pre_arr
    sub_arr = sub_arr - sub_arr.min()
    sub_arr = sub_arr.astype(np.uint16)
    
    sub_nii = sitk.GetImageFromArray(sub_arr)
    sub_nii.CopyInformation(pre_nii)
    
    output_path = path_patient / 'sub.nii.gz'
    sitk.WriteImage(sub_nii, str(output_path))
    logger.debug(f"Saved subtraction image to {output_path}")



if __name__ == "__main__":

    path_root = Path(r'\\rad-maid-004\D\PENN-MRI') 
    path_root_out = path_root / 'preprocessed'
    path_root_out_data = path_root_out / 'data'

    patient_dirs = [p for p in path_root_out_data.iterdir() if p.is_dir()]
    logger.info(f"Found {len(patient_dirs)} patient directories to process.")
    count = 0
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(process, patient_dirs), total=len(patient_dirs)):
            pass

    logger.info("Subtraction calculation complete.")

        
    