from pathlib import Path 
import logging
import torchio as tio 
import SimpleITK as sitk
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm


logger = logging.getLogger(__name__)


def process(path_patient):
    patient_id = path_patient.name
    logger.debug(f"Patient ID: {patient_id}")


    # Compute subtraction image
    logger.debug(f"Compute and write sub to disk")
    dyn0_nii = sitk.ReadImage(str(path_patient/'pre.nii.gz'), sitk.sitkInt32) # Note: if dtype not specified, data is read as uint16 -> subtraction wrong
    dyn1_nii_orig = sitk.ReadImage(str(path_patient/'post_1.nii.gz'), sitk.sitkInt32) # Renamed to avoid confusion
    
    logger.debug(f"Resampling post_1.nii.gz to match pre.nii.gz space.")
    dyn1_nii = sitk.Resample(dyn1_nii_orig, dyn0_nii, sitk.Transform(), sitk.sitkLinear, 0, dyn0_nii.GetPixelID())

    # Resample dyn1_nii_orig to the space of dyn0_nii
    logger.debug(f"Resampling post_1.nii.gz to match pre.nii.gz space.")
  #  dyn1_nii = sitk.Resample(dyn1_nii_orig, dyn0_nii, sitk.Transform(), sitk.sitkLinear, 0, dyn0_nii.GetPixelID())

    dyn0 = sitk.GetArrayFromImage(dyn0_nii)
    dyn1 = sitk.GetArrayFromImage(dyn1_nii) 
    sub = dyn0 - dyn1
    sub = sub - sub.min()
    #sub = np.clip(sub, 0, 65535) 
#    scale = (np.max(sub) - np.min(sub)) / 8
 #   sub = 65535 / (1 + np.exp(-sub / scale))
#    sub = np.clip(sub, 0,65535)  # Ensure no negative values
#    if sub.min() < 0:
#        sub = sub - sub.min()
    sub = sub.astype(np.uint16)
    sub_nii = sitk.GetImageFromArray(sub)
    sub_nii.CopyInformation(dyn0_nii)
    sitk.WriteImage(sub_nii, str(path_patient/'sub.nii.gz'))


    # Compute resampled T1-weighted image
    logger.debug(f"Compute and write resampled T1 to disk")
    t1_nii = sitk.ReadImage(str(path_patient/'T1.nii.gz'), sitk.sitkInt16)
    t1_resampled_nii = sitk.Resample(t1_nii, dyn0_nii, sitk.Transform(), sitk.sitkLinear, 0, dyn0_nii.GetPixelID()) # Interpolation: sitk.sitkBSpline, sitk.sitkLinear
    sitk.WriteImage(t1_resampled_nii, str(path_patient/'T1_resampled.nii.gz'))



if __name__ == "__main__":
    path_root =  Path(r'\\rad-maid-004\D\Duke-Cancer_MRI')
    path_root_out = path_root/'preprocessed-mst-with-seg'
    path_root_out_data = path_root_out/'data'

    files = list(path_root_out_data.iterdir())  # Convert the iterator to a list

    # Option 1: Multi-CPU 
    # with Pool(processes=4) as pool:
    #     for _ in tqdm(pool.imap_unordered(process, files), total=len(files)):
    #         pass

 #   Option 2: Single-CPU 
    for path_dir in tqdm(files):
        process(path_dir)
        
    