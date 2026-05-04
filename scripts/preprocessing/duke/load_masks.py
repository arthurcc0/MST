import matplotlib.pyplot as plt 
import numpy as np 
import nrrd
import SimpleITK as sitk

from pathlib import Path
import os
from tqdm import tqdm 

path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI') 

path_masks = path_root / 'PKG - Duke-Breast-Cancer-MRI-Supplement-v3'/ 'Duke-Breast-Cancer-MRI-Supplement-v3'/'Segmentation_Masks_NRRD'
save_dir = path_root / 'duke_masks'
save_dir.mkdir(exist_ok=True, parents=True)
def load_data():
    masks = list(path_masks.rglob('*_Breast.seg.nrrd'))
    for mask in tqdm(masks):
        data, _ = nrrd.read(mask)       
        out_name = mask.name.replace('_Breast.seg.nrrd', '.npy')
        np.save(save_dir / out_name, data)
        
    print("Done")

if __name__ == "__main__":
    load_data()
    
    
