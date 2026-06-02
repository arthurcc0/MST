import numpy as np
from pathlib import Path
from natsort import natsorted
import torchio as tio
from multiprocessing import Pool
from tqdm import tqdm
import functools
import torch
import nibabel as nib
from scipy.ndimage import rotate

from step2b_crop_or_pad import get_breast_crop_transform, TARGET_SHAPE

def process_mask(image_path, save_dir_str, rotate_img=False):
    """Loads a single .nii.gz image, splits it, crops each side, and saves them."""
    save_dir = Path(save_dir_str)
    try:
        # Load image data and affine using nibabel
        nii_img = nib.load(str(image_path))
        img_data = nii_img.get_fdata()
        affine = nii_img.affine

        # Rotate the image if requested
        if rotate_img and img_data.ndim == 3:
            # Rotate 90 degrees counter-clockwise along x and y axes
            img_data = rotate(img_data, angle=90, axes=(0, 1), reshape=False)

        # Add a channel dimension to the image data to make it 4D
        img_data = np.expand_dims(img_data, axis=0)

        # Create torchio subject from the potentially rotated numpy array
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_data, affine=affine)
        )

        # Resample to a standard orientation to ensure consistent splitting
        resample_transform = tio.ToCanonical()
        subject = resample_transform(subject)

        # Split along the width (first spatial dimension)
        width = subject.image.shape[1]
        split_transforms = {
            'right': tio.Crop((width // 2, 0, 0, 0, 0, 0)),
            'left': tio.Crop((0, width // 2, 0, 0, 0, 0)),
        }

        for side in ['left', 'right']:
            # Apply the split transform
            side_subject = split_transforms[side](subject)

            # Define transforms for this side
            crop_transform = get_breast_crop_transform(side_subject.image, target_height=TARGET_SHAPE[0])
            final_transform = tio.Compose([
                crop_transform,
                tio.CropOrPad(TARGET_SHAPE, padding_mode=0),
            ])

            # Apply the transform
            processed_subject = final_transform(side_subject)

            # Save the processed image with a side-specific name
            file_stem = image_path.name.replace('.nii.gz', '')
            output_path = save_dir / f'{file_stem}_{side}.nii.gz'
            processed_subject.image.save(output_path)

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

def crop_images():
    # The input directory now points to the masked images you created
    image_dir = Path(r'D:\PENN-MRI\pennmasked\data')
    # A new directory for the cropped output
    save_dir = Path(r'D:\PENN-MRI\penn_masked_cropped\data')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Look for .nii.gz files
    images_to_process = natsorted(list(image_dir.glob('*.nii.gz')))
    print(f"Found {len(images_to_process)} images to process.")

    # Use multiprocessing to speed up
    save_dir_str = str(save_dir)
    # The function to call is now process_mask, but we're processing images
    # Set rotate_img=True to ensure images are in the correct orientation for cropping.
    partial_process = functools.partial(process_mask, save_dir_str=save_dir_str, rotate_img=True)

    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial_process, images_to_process), total=len(images_to_process)):
            pass

    print("Finished cropping all images.")

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent))
    crop_images()
    # Option 2: Single-CPU 
    # for path_dir in tqdm(path_patients):
    #     preprocess(path_dir)     

