import torch
import torchio as tio
import numpy as np
from pathlib import Path
from natsort import natsorted
from multiprocessing import Pool
from tqdm import tqdm
import functools
import nibabel as nib
from scipy.ndimage import rotate
import sys
import re

# Add parent directory to path to import cropping logic
sys.path.append(str(Path(__file__).parent))
from step2b_crop_or_pad import get_breast_crop_transform, TARGET_SHAPE

def process_and_mask(image_path, mask_dict, save_dir_str):
    """
    Loads an original image and its corresponding mask, applies identical cropping and rotation
    to both, multiplies them, and saves the final result.
    """
    save_dir = Path(save_dir_str)
    image_stem = image_path.stem

    try:
        mask_path = mask_dict.get(image_stem)
        if not mask_path:
            return

        # Load original image and mask data
        image_data = np.load(image_path, allow_pickle=True)
        mask_data = np.load(mask_path, allow_pickle=True)

        if image_data.size == 0 or mask_data.size == 0:
            return

        # --- Consistently process both image and mask ---
        # Create a single TorchIO subject with image and mask
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(image_data).unsqueeze(0)),
            mask=tio.LabelMap(tensor=torch.from_numpy(mask_data).unsqueeze(0)),
        )

        # Reorient to canonical first
        reorient_transform = tio.ToCanonical()
        subject = reorient_transform(subject)

        # Rotate 90 degrees, as required for cropping logic
        image_rotated = rotate(subject.image.data.squeeze().numpy(), angle=90, axes=(0, 1), reshape=False)
        mask_rotated = rotate(subject.mask.data.squeeze().numpy(), angle=90, axes=(0, 1), reshape=False)
        
        # Recreate subject with rotated data but original affine
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(image_rotated).unsqueeze(0), affine=subject.image.affine),
            mask=tio.LabelMap(tensor=torch.from_numpy(mask_rotated).unsqueeze(0), affine=subject.mask.affine),
        )

        # Split into left and right sides
        width = subject.spatial_shape[1] # Width is the second spatial dimension after reorientation
        split_transforms = {
            'right': tio.Crop((0, width // 2, 0, 0, 0, 0)),
            'left': tio.Crop((width // 2, 0, 0, 0, 0, 0)),
        }

        for side in ['left', 'right']:
            side_subject = split_transforms[side](subject)

            # Get the same cropping transform for both based on the image
            crop_transform = get_breast_crop_transform(side_subject.image, target_height=TARGET_SHAPE[0])
            pad_transform = tio.CropOrPad(TARGET_SHAPE, padding_mode=0)
            
            final_transform = tio.Compose([crop_transform, pad_transform])

            # Apply the exact same transform to the subject (image and mask)
            processed_subject = final_transform(side_subject)

            # Get the final cropped image and mask data
            final_image_data = processed_subject.image.data.squeeze().numpy()
            final_mask_data = processed_subject.mask.data.squeeze().numpy()

            # Apply the final mask
            masked_output = final_image_data * final_mask_data

            # Extract patient ID from the filename
            patient_id = image_stem.split('_')[0]

            # Save the result, preserving the affine transformation for correct orientation
            output_dir = save_dir / f'{patient_id}_{side}'
            output_dir.mkdir(parents=True, exist_ok=True)

            output_image = tio.ScalarImage(tensor=torch.from_numpy(masked_output).unsqueeze(0), affine=processed_subject.image.affine)
            output_path = output_dir / 'sub.nii.gz'
            output_image.save(output_path)

    except Exception as e:
        print(f"Error processing {image_stem}: {e}")

def main():
    # Original, unmasked images (from network drive)
    image_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI\output\preproc\n4bc_plhe')
    # Original segmentation masks (from network drive)
    mask_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI\output\breast')
    # Final output directory
    save_dir = Path(r'D:\PENN-MRI\penn-preprocessed_cropped2\data')
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image file list from cache...")
    cache_file = Path(__file__).parent / 'image_file_cache.txt'
    if not cache_file.exists():
        print("Error: image_file_cache.txt not found. Please run the caching script first.")
        return

    with open(cache_file, 'r') as f:
        all_image_paths = [Path(line.strip()) for line in f.readlines()]

    masks = natsorted(list(mask_dir.glob('*.npy')))
    mask_dict = {mask.stem: mask for mask in masks}

    # Filter for images that have a corresponding mask
    images_to_process = [p for p in all_image_paths if p.stem in mask_dict]
    print(f"Found {len(images_to_process)} images with matching masks to process.")

    partial_process = functools.partial(process_and_mask, mask_dict=mask_dict, save_dir_str=str(save_dir))

    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial_process, images_to_process), total=len(images_to_process)):
            pass

    print(f"Finished processing all images. Results are in {save_dir}")

if __name__ == "__main__":
    main()
