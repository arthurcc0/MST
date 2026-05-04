from pathlib import Path
import torchio as tio
import torch
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import functools

def crop_breast_height(image, margin_top=10):
    """Crop height to 256 and try to cover breast based on intensity localization."""
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    if torch.any(fg_rows):
        top = min(max(512 - int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    else:
        top = 128  # Default crop if no foreground found
    bottom = 256 - top
    return tio.Crop((0, 0, bottom, top, 0, 0))

def preprocess_mask(path_dir, path_root_in_data_str, path_root_out_data_str):
    """Preprocesses a single mask by applying transformations derived from a reference image."""
    path_root_in_data = Path(path_root_in_data_str)
    path_root_out_data = Path(path_root_out_data_str)

    path_mask = path_dir / 'mask_1.nii.gz'
    path_ref_img = path_dir / 'pre.nii.gz'

    if not path_mask.exists() or not path_ref_img.exists():
        return

    # Load images
    ref_img = tio.ScalarImage(path_ref_img)
    mask = tio.LabelMap(path_mask)

    # Define transforms based on the reference image
    target_spacing = (0.7, 0.7, 3)
    target_shape = (512, 512, 32)

    # Create a temporary transformed ref_img to calculate the height crop
    temp_transform = tio.Compose([
        tio.Resample(target_spacing),
        tio.CropOrPad(target_shape, padding_mode=0),
        tio.ToCanonical(),
    ])
    transformed_ref_img = temp_transform(ref_img)
    crop_height_transform = crop_breast_height(transformed_ref_img)

    # Define the full transform pipeline for the mask
    full_transform = tio.Compose([
        tio.Resample(target_spacing, image_interpolation='nearest'),
        tio.CropOrPad(target_shape, padding_mode=0),
        tio.ToCanonical(),
        crop_height_transform,
    ])

    # Apply transforms to the mask
    transformed_mask = full_transform(mask)

    # Split and save
    split_side = {
        'right': tio.Crop((256, 0, 0, 0, 0, 0)),
        'left': tio.Crop((0, 256, 0, 0, 0, 0)),
    }

    for side in ['left', 'right']:
        path_out_dir = path_root_out_data / f"{path_dir.relative_to(path_root_in_data)}_{side}"
        path_out_dir.mkdir(exist_ok=True, parents=True)

        mask_side = split_side[side](transformed_mask)
        mask_side.save(path_out_dir / path_mask.name)

if __name__ == "__main__":
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI')
    path_root_in_data = path_root / 'preprocessed-mst-with-seg' / 'data'
    path_root_out = path_root / 'preprocessed_crop-with-seg'
    path_root_out_data = path_root_out / 'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    path_patients = [p for p in path_root_in_data.iterdir() if p.is_dir() and (p / 'mask_1.nii.gz').exists()]

    path_root_in_data_str = str(path_root_in_data)
    path_root_out_data_str = str(path_root_out_data)

    partial_preprocess = functools.partial(preprocess_mask,
                                           path_root_in_data_str=path_root_in_data_str,
                                           path_root_out_data_str=path_root_out_data_str)

    # Option 1: Multi-CPU
    with Pool(processes=4) as pool:
        for _ in tqdm(pool.imap_unordered(partial_preprocess, path_patients), total=len(path_patients)):
            pass

    # Option 2: Single-CPU
    # for path_dir in tqdm(path_patients):
    #     preprocess_mask(path_dir, path_root_in_data_str, path_root_out_data_str)

