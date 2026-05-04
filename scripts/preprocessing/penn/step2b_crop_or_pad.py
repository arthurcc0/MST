from pathlib import Path 
import torchio as tio 
import torch
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm
import functools


def crop_breast_height(image):
    "Crop height to 256 and try to cover breast based on intensity localization"
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    
    image_height = image.shape[2]
    target_height = 256

    # Get breast boundaries
    fg_indices = torch.argwhere(fg_rows)
    if len(fg_indices) == 0:
        # No foreground found, crop from center
        top_crop = (image_height - target_height) // 2
        bottom_crop = image_height - target_height - top_crop
        return tio.Crop((0, 0, bottom_crop, top_crop, 0, 0))

    breast_top_row = fg_indices.min().item()
    breast_bottom_row = fg_indices.max().item()
    breast_center = (breast_top_row + breast_bottom_row) // 2

    # Calculate crop window
    half_height = target_height // 2
    crop_start = breast_center - half_height
    crop_end = breast_center + half_height

    # Adjust window if it's out of bounds
    if crop_start < 0:
        crop_end -= crop_start
        crop_start = 0
    if crop_end > image_height:
        crop_start -= (crop_end - image_height)
        crop_end = image_height
    
    # Ensure start is not negative after adjustment
    crop_start = max(0, crop_start)

    # Calculate cropping values for TorchIO (pixels to remove)
    top_crop = crop_start
    bottom_crop = image_height - crop_end

    return tio.Crop((0, 0, bottom_crop, top_crop, 0, 0))


def preprocess(path_dir, path_root_in_data_str, path_root_out_data_str):
    path_root_in_data = Path(path_root_in_data_str)
    path_root_out_data = Path(path_root_out_data_str)
    # -------- Settings --------------
    ref_img = tio.ScalarImage(path_dir/'pre.nii.gz')
    # Set target shape and spacing
    target_spacing = (1.0, 1.0, 1.0) 
    target_shape = (512, 512, 32)
    ref_img = tio.Resample(target_spacing)(ref_img)


    transform = tio.Compose([
        tio.Resample(ref_img), # Resample to reference image to ensure that origin, direction, etc, fit
        tio.ToCanonical(),
    ])

    target_shape_side = (256, 256, 32)

    for n, path_img in enumerate(path_dir.glob('*.nii.gz')):
        # Read image 
        img = tio.ScalarImage(path_img)

        # Skip volumes with x-dimension less than 100
        if img.shape[1] < 100:
            print(f"Skipping {path_img.name} due to small x-dimension: {img.shape[1]}")
            continue

        # Apply initial transforms
        img = transform(img)

        # Split left and right side 
        width = img.shape[1]
        split_side = {
            'right': tio.Crop((width // 2, 0, 0, 0, 0, 0)),
            'left': tio.Crop((0, width // 2, 0, 0, 0, 0)),
        }

        for side in ['left', 'right']:
            # Create output directory 
            path_out_dir = path_root_out_data/f"{path_dir.relative_to(path_root_in_data)}_{side}"
            path_out_dir.mkdir(exist_ok=True, parents=True)

            # Crop left/right side 
            img_side = split_side[side](img)

            # Pad and crop each side individually
            side_transform = tio.Compose([
                crop_breast_height(img_side),
                tio.CropOrPad(target_shape_side, padding_mode=0),
            ])
            img_side = side_transform(img_side)

            # Save 
            img_side.save(path_out_dir/path_img.name)

if __name__ == "__main__":
    path_root = Path(r'\\rad-maid-004\D\PENN-MRI')
    path_root_in_data = path_root / 'preprocessed' / 'data'
    path_root_out = path_root / 'preprocessed_cropped'
    path_root_out_data = path_root_out / 'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    path_patients = [p for p in path_root_in_data.iterdir() if p.is_dir()]
    
    # Option 1: Multi-CPU 
    # Convert Path objects to strings for serialization
    path_root_in_data_str = str(path_root_in_data)
    path_root_out_data_str = str(path_root_out_data)
    
    partial_preprocess = functools.partial(preprocess, 
                                           path_root_in_data_str=path_root_in_data_str, 
                                           path_root_out_data_str=path_root_out_data_str)
    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(partial_preprocess, path_patients), total=len(path_patients)):
            pass

    # Option 2: Single-CPU 
    # for path_dir in tqdm(path_patients):
    #     preprocess(path_dir)
    