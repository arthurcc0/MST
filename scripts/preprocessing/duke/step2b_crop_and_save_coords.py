import argparse
from pathlib import Path
import torchio as tio
import torch
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import functools
import json

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
    """Processes a single patient directory to save cropping coordinates."""
    path_root_in_data = Path(path_root_in_data_str)
    path_root_out_data = Path(path_root_out_data_str)
    
    ref_img_path = path_dir / 'pre.nii.gz'
    if not ref_img_path.exists():
        print(f"Skipping {path_dir.name}: 'pre.nii.gz' not found.")
        return

    ref_img = tio.ScalarImage(ref_img_path)
    original_shape = ref_img.spatial_shape
    original_spacing = ref_img.spacing
    original_affine = ref_img.affine.tolist()

    target_spacing = (0.7, 0.7, 3)
    target_shape = (512, 512, 32)
    
    resample_transform = tio.Compose([
        tio.Resample(target_spacing),
        tio.ToCanonical(),
    ])
    resampled_img = resample_transform(ref_img)
    resampled_shape = resampled_img.spatial_shape

    pad_transform = tio.CropOrPad(target_shape, padding_mode=0)
    padded_img = pad_transform(resampled_img)

    crop_height = crop_breast_height(padded_img)
    cropping_params = crop_height.cropping
    
    pre_crop_shape = padded_img.spatial_shape

    coords = {
        'height_crop': {
            'bottom': cropping_params[2],
            'top': cropping_params[3]
        },
        'reconstruction_shapes': {
            'original': [int(s) for s in original_shape],
            'original_spacing': [float(s) for s in original_spacing],
            'original_affine': original_affine,
            'after_resample': [int(s) for s in resampled_shape],
            'after_pad': [int(s) for s in pre_crop_shape],
            'pre_split': [int(pre_crop_shape[0]), 256, int(pre_crop_shape[2])],
            'full_reconstructed': [int(s) for s in original_shape]
        }
    }

    # Only create directories and save coordinates file.
    for side in ['left', 'right']:
        path_out_dir = path_root_out_data / f"{path_dir.relative_to(path_root_in_data)}_{side}"
        path_out_dir.mkdir(exist_ok=True, parents=True)

        with open(path_out_dir / 'cropping_coordinates.json', 'w') as f:
            json.dump(coords, f, indent=4)

def main():
    """Main function to parse arguments and run the coordinate saving."""
    parser = argparse.ArgumentParser(description="Calculate and save cropping coordinates for breast MRI processing.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing patient folders with NIfTI files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the coordinate files.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for processing.')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of patients to process (0 for no limit).')
    args = parser.parse_args()

    path_root_in_data = Path(args.input_dir)
    path_root_out_data = Path(args.output_dir)
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    path_patients = [p for p in path_root_in_data.iterdir() if p.is_dir()]

    if args.limit > 0:
        path_patients = path_patients[:args.limit]
        print(f"Processing a limit of {len(path_patients)} patients.")
    else:
        print(f"Found {len(path_patients)} patient directories to process.")

    path_root_in_data_str = str(path_root_in_data)
    path_root_out_data_str = str(path_root_out_data)

    partial_preprocess = functools.partial(preprocess,
                                           path_root_in_data_str=path_root_in_data_str,
                                           path_root_out_data_str=path_root_out_data_str)
    
    if args.num_workers > 1 and len(path_patients) > 1:
        print(f"Starting coordinate generation with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            list(tqdm(pool.imap_unordered(partial_preprocess, path_patients), total=len(path_patients)))
    else:
        print("Starting coordinate generation with a single worker...")
        for path_dir in tqdm(path_patients):
            preprocess(path_dir, path_root_in_data_str, path_root_out_data_str)
    
    print("Coordinate generation complete.")

if __name__ == "__main__":
    main()
