from pathlib import Path 
import torchio as tio 
import torch
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm
import functools


def crop_breast_height(image, margin_top=10):
    """Crop height to 256 and try to cover breast based on intensity localization"""
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    
    # Check if there is any foreground to avoid errors on empty images
    fg_indices = torch.argwhere(fg_rows)
    if fg_indices.numel() > 0:
        top = min(max(512 - int(fg_indices.max()) - margin_top, 0), 256)
    else:
        # Default cropping if no foreground is found (e.g., blank image)
        top = 128
    
    bottom = 256 - top
    return tio.Crop((0, 0, bottom, top, 0, 0))
def preprocess(path_dir, path_root_in_data_str, path_root_out_data_str):
    path_root_in_data = Path(path_root_in_data_str)
    path_root_out_data = Path(path_root_out_data_str)
    
    # -------- Settings --------------
    ref_img = tio.ScalarImage(path_dir/'pre.nii.gz')

    # Option: Static 
    target_spacing = (0.7, 0.7, 3) 
    target_shape = (512, 512, 32)
    ref_img = tio.Resample(target_spacing)(ref_img)

    transform = tio.Compose([
        tio.Resample(ref_img), # Resample to reference image to ensure that origin, direction, etc, fit
        tio.CropOrPad(target_shape, padding_mode=0),
        tio.ToCanonical(),
    ])
    crop_height = crop_breast_height(transform(ref_img))     
    split_side = {
        'right': tio.Crop((256, 0, 0, 0, 0, 0)),
        'left': tio.Crop((0, 256, 0, 0, 0, 0)),
    }
    
    for n, path_img in enumerate(path_dir.glob('*.nii.gz')):
        # Read image 
        img = tio.ScalarImage(path_img)

        # Preprocess (eg. Crop/Pad)
        img = transform(img)

        # Crop bottom and top so that height is 256 and breast is preserved  
        img = crop_height(img)

        # Split left and right side 
        for side in ['left', 'right']:
            # Create output directory 
            path_out_dir = path_root_out_data/f"{path_dir.relative_to(path_root_in_data)}_{side}"
            path_out_dir.mkdir(exist_ok=True, parents=True)

            # Crop left/right side 
            img_side = split_side[side](img)

            # Save 
            img_side.save(path_out_dir/path_img.name)


if __name__ == "__main__":
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI')
    path_root_in_data = path_root/'preprocessed-mst-a'/'data'  # Input directory containing patient folders with pre.nii.gz, post_1.nii.gz, T1.nii.gz
    path_root_out = path_root/'preprocessed_crop-a'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    
    path_patients = list(path_root_in_data.iterdir())  # Convert the iterator to a list
    
    # Option 1: Multi-CPU 
    # Convert Path objects to strings for serialization
    path_root_in_data_str = str(path_root_in_data)
    path_root_out_data_str = str(path_root_out_data)
    
    partial_preprocess = functools.partial(preprocess, 
                                           path_root_in_data_str=path_root_in_data_str, 
                                           path_root_out_data_str=path_root_out_data_str)
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(partial_preprocess, path_patients), total=len(path_patients)):
            pass

    # Option 2: Single-CPU 
    # for path_dir in tqdm(path_patients):
    #     preprocess(path_dir)
        
    