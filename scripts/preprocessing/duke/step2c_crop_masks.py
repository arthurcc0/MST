from pathlib import Path 
import numpy as np
import torchio as tio 
import torch
from multiprocessing import Pool
from tqdm import tqdm
import functools


def crop_breast_height(image, margin_top=10):
    """Crop height to 256 and try to cover breast based on intensity localization"""
    # threshold = int(image.data.float().quantile(0.9))
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    # Check if there are any foreground pixels
    fg_indices = torch.argwhere(fg_rows)
    if len(fg_indices) > 0:
        top = min(max(512-int(fg_indices.max()) - margin_top, 0), 256)
    else:
        # If no foreground pixels found, use default cropping from the top
        top = 256
    bottom = 256-top
    return tio.Crop((0, 0, bottom, top, 0, 0))


def preprocess_mask(mask_path, path_root_out_data_str):
    """Process a single mask file"""
    path_root_out_data = Path(path_root_out_data_str)
    
    # Load mask as numpy array
    mask_np = np.load(mask_path)
    mask_np = np.rot90(mask_np, k=-1, axes=(0, 1)).copy()
    
    # Convert numpy array to TorchIO ScalarImage
    # Note: TorchIO expects data in format (C, H, W, D) where C is channels
    if mask_np.ndim == 3:
        mask_np = mask_np[np.newaxis, ...]  # Add channel dimension
    
    # Create TorchIO image from numpy array
    mask_img = tio.ScalarImage(tensor=torch.from_numpy(mask_np).float())
    
    # Get reference spacing and shape (similar to step2b_crop_or_pad.py)
    target_spacing = list(mask_img.spacing) 
    target_spacing[-1] = 3
    mask_img = tio.Resample(target_spacing)(mask_img)
    target_shape = (512, 512, 32)

    # Apply transformations
    transform = tio.Compose([
        tio.CropOrPad(target_shape, padding_mode=0),
        tio.ToCanonical(),
    ])
    
    # Apply transform
    mask_img = transform(mask_img)
    
    # Crop height based on breast localization
    crop_height = crop_breast_height(mask_img)     
    mask_img = crop_height(mask_img)
    
    # Split left and right side 
    split_side = {
        'right': tio.Crop((256, 0, 0, 0, 0, 0)),
        'left': tio.Crop((0, 256, 0, 0, 0, 0)),
    }
    
    # Get mask filename without extension
    mask_name = mask_path.stem
    
    # Split left and right side and save
    for side in ['left', 'right']:
        # Create output directory 
        path_out_dir = path_root_out_data / f"{mask_name}_{side}"
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Crop left/right side 
        mask_side = split_side[side](mask_img)

        # Save as .nii.gz (to match the format expected by other scripts)
        mask_side.save(path_out_dir / f"{mask_name}.nii.gz")


if __name__ == "__main__":
    # Input directory containing mask files
    path_masks_dir = Path(r'D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\temp-breast')
    
    # Output directory for cropped masks
    path_root_out = Path(r'D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\cropped-masks')
    path_root_out_data = path_root_out / 'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    # Get all .npy mask files
    mask_files = list(path_masks_dir.glob('*.npy'))
    print(f"Found {len(mask_files)} mask files to process")
    
    # Option 1: Multi-CPU processing
    # Convert Path objects to strings for serialization
    path_root_out_data_str = str(path_root_out_data)
    
    partial_preprocess = functools.partial(preprocess_mask, 
                                           path_root_out_data_str=path_root_out_data_str)
    
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(partial_preprocess, mask_files), total=len(mask_files)):
            pass

    # Option 2: Single-CPU (uncomment if needed)
    # for mask_file in tqdm(mask_files):
    #     preprocess_mask(mask_file, path_root_out_data_str)
        
    print("Mask cropping completed!")
