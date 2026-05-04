import numpy as np
import SimpleITK as sitk
from pathlib import Path
from natsort import natsorted
import torchio as tio
import torch
from multiprocessing import Pool
from tqdm import tqdm
import functools
import os
import re

def process_image(image_path, mask_dict, save_dir_str, reorient_transform):
    """Processes a single image: reorients, masks, and saves it."""
    save_dir = Path(save_dir_str)
    image_stem = image_path.stem
    try:
        mask_path = mask_dict.get(image_stem)
        if not mask_path:
            return

        # Load the image and mask
        image_data = np.load(image_path, allow_pickle=True)
        mask_data = np.load(mask_path, allow_pickle=True)

        if image_data.size == 0:
            print(f"Skipping {image_path.name}: Image file is empty.")
            return
        if mask_data.size == 0:
            print(f"Skipping {mask_path.name}: Mask file is empty.")
            return

        # Create TorchIO Subjects
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(image_data).unsqueeze(0)),
            mask=tio.LabelMap(tensor=torch.from_numpy(mask_data).unsqueeze(0)),
        )

        subject_reoriented = reorient_transform(subject)
        image_axial = subject_reoriented.image.data.squeeze().numpy()
        mask_axial = subject_reoriented.mask.data.squeeze().numpy()
        masked_data = image_axial * mask_axial

        masked_image_tio = tio.ScalarImage(tensor=torch.from_numpy(masked_data).unsqueeze(0), affine=subject_reoriented.image.affine)
        
        # Sanitize the filename to remove invalid characters
        sanitized_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', image_stem)
        output_path = Path(save_dir_str) / f'{sanitized_stem}.nii.gz'
        
        masked_image_tio.save(output_path)

    except ValueError as e:
        # Catch errors from empty or malformed .npy files
        print(f"Skipping {image_stem}.npy due to ValueError: {e}")
    except Exception as e:
        # Catch other errors, like file access issues
        print(f"Failed to process {image_stem}.npy: {e}")

def apply_mask_parallel():
    mask_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI\output\breast')
    save_dir = Path(r'D:\PENN-MRI\pennmasked\data')
    save_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI\output\preproc\n4bc_plhe')

    # Find all masks first
    masks = natsorted(list(mask_dir.glob('*.npy')))
    print(f'Found {len(masks)} total masks.')

    # Load the list of image paths from the local cache file.
    cache_file = Path(__file__).parent / 'image_file_cache.txt'
    if not cache_file.exists():
        print(f"Error: Cache file not found at {cache_file}")
        print("Please run 'create_file_list_cache.py' first to generate the file list.")
        return

    print(f"Loading image file list from cache: {cache_file}")
    with open(cache_file, 'r') as f:
        all_image_paths = [Path(line.strip()) for line in f.readlines()]
    print(f"Loaded {len(all_image_paths)} image paths from cache.")

    # Create a dictionary of images for fast lookup
    image_dict = {p.stem: p for p in all_image_paths}

    # Match masks to images
    images_to_process = []
    mask_dict = {}
    for mask_path in masks:
        if mask_path.stem in image_dict:
            images_to_process.append(image_dict[mask_path.stem])
            mask_dict[mask_path.stem] = mask_path

    print(f'Found {len(images_to_process)} images with a matching mask. Starting processing...')

    reorient_transform = tio.ToCanonical()
    
    # Use functools.partial to prepare the function for the pool
    partial_process = functools.partial(
        process_image, 
        mask_dict=mask_dict, 
        save_dir_str=str(save_dir), 
        reorient_transform=reorient_transform
    )

    # Process only the matched images in parallel
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial_process, images_to_process), total=len(images_to_process)):
            pass

    print("Finished applying masks to all images.")

if __name__ == '__main__':
    apply_mask_parallel()