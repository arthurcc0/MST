import torch
import torchio as tio
import numpy as np
from pathlib import Path
from natsort import natsorted
from multiprocessing import Pool
from tqdm import tqdm
import functools
from scipy.ndimage import rotate
import sys
import re
import nibabel as nib

# Add parent directory to path to import cropping logic
sys.path.append(str(Path(__file__).parent))
from step2b_crop_or_pad import get_breast_crop_transform

def process_pair(image_path, original_mask_dict, save_dir_str):
    """
    Loads a preprocessed image and its original mask, applies identical cropping to both,
    multiplies them, and saves the result.
    """
    save_dir = Path(save_dir_str)
    
    try:
        # The patient ID is the name of the parent directory of the image file
        patient_id = image_path.parent.name

        # Find the corresponding original mask
        mask_path = original_mask_dict.get(patient_id)
        if not mask_path:
            # print(f"Warning: No mask found for {patient_id}")
            return

        # --- Load Data ---
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata().astype(np.float32)
        mask_data = np.load(mask_path, allow_pickle=True).astype(np.float32)

        if image_data.size == 0 or mask_data.size == 0:
            return

        # --- Process Image and Mask Separately Before Combining ---
        reorient_transform = tio.ToCanonical()

        # Process image: reorient only
        image_subject = tio.Subject(image=tio.ScalarImage(tensor=torch.from_numpy(image_data).unsqueeze(0), affine=image_nii.affine))
        image_reoriented = reorient_transform(image_subject)

        # Process mask: reorient AND rotate
        mask_subject = tio.Subject(mask=tio.LabelMap(tensor=torch.from_numpy(mask_data).unsqueeze(0), affine=image_nii.affine))
        mask_reoriented = reorient_transform(mask_subject)
        mask_rotated_data = rotate(mask_reoriented.mask.data.squeeze().numpy(), angle=90, axes=(0, 1), reshape=False)

        # --- Create a new subject with the processed data ---
        # The image is reoriented; the mask is reoriented and rotated.
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_reoriented.image.data, affine=image_reoriented.image.affine),
            mask=tio.LabelMap(tensor=torch.from_numpy(mask_rotated_data).unsqueeze(0), affine=image_reoriented.image.affine),
        )

        # 3. Split into left and right sides
        width = subject.spatial_shape[1]
        split_transforms = {
            'right': tio.Crop((0, width // 2, 0, 0, 0, 0)),
            'left': tio.Crop((width // 2, 0, 0, 0, 0, 0)),
        }

        for side in ['left', 'right']:
            side_subject = split_transforms[side](subject)

            # 4. Get cropping and padding transforms
            crop_transform = get_breast_crop_transform(side_subject.image, target_height=256)
            pad_transform = tio.CropOrPad((256, 256, 32), padding_mode=0)
            final_transform = tio.Compose([crop_transform, pad_transform])

            processed_subject = final_transform(side_subject)

            # --- Apply Mask and Save ---
            final_image_data = processed_subject.image.data.squeeze().numpy()
            final_mask_data = processed_subject.mask.data.squeeze().numpy()
            masked_output = final_image_data * final_mask_data

            # Save the result
            output_image = tio.ScalarImage(tensor=torch.from_numpy(masked_output).unsqueeze(0), affine=processed_subject.image.affine)
            
            # Create a directory for the patient and side
            output_dir = save_dir / f'{patient_id}_{side}'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the file as sub.nii.gz
            output_path = output_dir / 'sub.nii.gz'
            output_image.save(output_path)

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

def main():
    # Input directories
    image_dir = Path(r'D:\PENN-MRI\penn-preprocessed2\data')
    mask_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI\output\breast')
    
    # Output directory
    save_dir = Path(r'D:\PENN-MRI\final_cropped_and_masked_data')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {save_dir}")

    # Load file paths, considering only '*sub.nii.gz' files
    images_to_process = natsorted(list(image_dir.glob('**/*sub.nii.gz')))
    masks = natsorted(list(mask_dir.glob('*.npy')))
    mask_dict = {mask.stem: mask for mask in masks}

    print(f"Found {len(images_to_process)} images and {len(masks)} masks.")

    # Use multiprocessing
    partial_process = functools.partial(process_pair, original_mask_dict=mask_dict, save_dir_str=str(save_dir))

    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial_process, images_to_process), total=len(images_to_process)):
            pass

    print("Finished processing all images.")

if __name__ == "__main__":
    main()
