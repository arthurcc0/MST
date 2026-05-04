import torch
import torchio as tio
from torchio.transforms import ToCanonical
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import functools

def reconstruct_patient(patient_id, coords_dir, seg_dir, out_dir):
    """Reconstructs a single patient's segmentation using coordinate and segmentation files."""
    try:
        # Extract numeric ID from patient_id like 'Breast_MRI_002' -> '2'
        numeric_id = patient_id.split('_')[-1].lstrip('0')
    except (IndexError, AttributeError):
        print(f"Skipping {patient_id}: could not parse numeric ID from patient ID.")
        return

    # Define paths for segmentation files
    left_seg_path = seg_dir / f"{numeric_id}_left_segmentation.nii.gz"
    right_seg_path = seg_dir / f"{numeric_id}_right_segmentation.nii.gz"

    # Define path for the coordinates file
    # A single coordinates file is stored for both left/right sides
    coord_folder_path = coords_dir / f"{patient_id}_left"
    coords_path = coord_folder_path / 'cropping_coordinates.json'

    if not left_seg_path.exists():
        print(f"Skipping {patient_id}: Left segmentation not found at {left_seg_path}")
        return
    if not right_seg_path.exists():
        print(f"Skipping {patient_id}: Right segmentation not found at {right_seg_path}")
        return
    if not coords_path.exists():
        print(f"Skipping {patient_id}: Coordinates not found at {coords_path}")
        return

    try:
        with open(coords_path, 'r') as f:
            coords = json.load(f)
        
        # Print JSON configuration for debugging
        print(f"\n[DEBUG] {patient_id} - JSON Configuration:")
        print(f"  - height_crop: top={coords['height_crop']['top']}, bottom={coords['height_crop']['bottom']}")
        if 'reconstruction_shapes' in coords:
            pre_split = coords.get('reconstruction_shapes', {}).get('pre_split', [])
            full_reconstructed = coords.get('reconstruction_shapes', {}).get('full_reconstructed', [])
            print(f"  - reconstruction_shapes: pre_split={pre_split}, full_reconstructed={full_reconstructed}")

        left_img = tio.ScalarImage(left_seg_path)
        right_img = tio.ScalarImage(right_seg_path)
        
        # Print dimensions of input images
        print(f"[DEBUG] {patient_id} - Input dimensions:")
        print(f"  - Left image: shape={left_img.data.shape}, dtype={left_img.data.dtype}")
        print(f"  - Right image: shape={right_img.data.shape}, dtype={right_img.data.dtype}")
        
        # Get expected dimensions from coordinate file
        expected_shape = None
        if 'reconstruction_shapes' in coords and 'pre_split' in coords['reconstruction_shapes']:
            expected_shape = coords['reconstruction_shapes']['pre_split']
            print(f"[DEBUG] {patient_id} - Expected pre_split shape from coords: {expected_shape}")
            
        # TorchIO loads as (C, D, H, W), but needs to be (C, W, H, D) for this script's logic
        # Permute dimensions to align with expected format
        left_data = left_img.data.permute(0, 3, 2, 1)
        right_data = right_img.data.permute(0, 3, 2, 1)
        
        print(f"[DEBUG] {patient_id} - After permute: left={left_data.shape}, right={right_data.shape}")
        
        # Check if resizing is needed to match expected dimensions
        if expected_shape:
            expected_height = expected_shape[1]  # Height is the second dimension in pre_split
            expected_width_half = expected_shape[0] // 2  # Width is the first dimension, divided by 2 for each half
            
            current_height = left_data.shape[2]
            current_width = left_data.shape[1]
            
            if current_height != expected_height or current_width != expected_width_half:
                print(f"[DEBUG] {patient_id} - Resizing needed: current_height={current_height}, expected_height={expected_height}, current_width={current_width}, expected_width_half={expected_width_half}")
                
                # Option 1: Resize using interpolation
                # Uncomment if you want to resize images
                # left_data = torch.nn.functional.interpolate(left_data, size=(expected_width_half, expected_height, left_data.shape[3]), mode='nearest')
                # right_data = torch.nn.functional.interpolate(right_data, size=(expected_width_half, expected_height, right_data.shape[3]), mode='nearest')
                
                # Option 2: Pad to expected dimensions
                if current_height < expected_height or current_width < expected_width_half:
                    pad_height = max(0, expected_height - current_height)
                    pad_width = max(0, expected_width_half - current_width)
                    
                    # Pad format: (left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad)
                    padding = (0, 0, 0, pad_height, 0, pad_width)
                    
                    print(f"[DEBUG] {patient_id} - Padding with: {padding}")
                    pad_layer = torch.nn.ConstantPad3d(padding, 0)
                    left_data = pad_layer(left_data)
                    right_data = pad_layer(right_data)
        
        # We concatenate along width (dim=1)
        combined_data = torch.cat((left_data, right_data), dim=1)
        
        # Create new combined image with the permuted data
        combined_img = tio.ScalarImage(tensor=combined_data, affine=left_img.affine)
        
        print(f"[DEBUG] {patient_id} - After concatenation: shape={combined_data.shape}")

        # Get cropping parameters
        top_crop = coords['height_crop']['top']
        bottom_crop = coords['height_crop']['bottom']

        # Define the padding to reverse the crop
        # tio.Crop format is (left, right, bottom, top, back, front)
        padding_params = (0, 0, bottom_crop, top_crop, 0, 0)
        pad_transform = tio.Pad(padding_params, padding_mode=0)

        # Apply the reverse padding
        reconstructed_img = pad_transform(combined_img)
        print(f"[DEBUG] {patient_id} - After padding: shape={reconstructed_img.data.shape}")

        # First, crop back to the size before the main padding was applied
        final_img = reconstructed_img
        if 'reconstruction_shapes' in coords and 'after_resample' in coords['reconstruction_shapes']:
            target_shape = coords['reconstruction_shapes']['after_resample']
            print(f"[DEBUG] {patient_id} - Cropping to resampled shape: {target_shape}")
            cropper = tio.CropOrPad(target_shape, padding_mode=0)
            final_img = cropper(reconstructed_img)
        
        # Now, resample back to original spacing and crop to original shape
        if 'reconstruction_shapes' in coords and 'original_spacing' in coords['reconstruction_shapes']:
            original_spacing = coords['reconstruction_shapes']['original_spacing']
            original_shape = coords['reconstruction_shapes']['original']
            
            print(f"[DEBUG] {patient_id} - Resampling back to spacing: {original_spacing}")
            inverse_resampler = tio.Resample(target=original_spacing, image_interpolation='nearest')
            resampled_to_original_img = inverse_resampler(final_img)

            print(f"[DEBUG] {patient_id} - Cropping to final original shape: {original_shape}")
            final_cropper = tio.CropOrPad(target_shape=original_shape, padding_mode=0)
            final_img = final_cropper(resampled_to_original_img)
        else:
            # Fallback for old coordinate files
            print(f"[WARNING] {patient_id} - 'original_spacing' not in coordinates file. Skipping final resampling.")
            if 'reconstruction_shapes' in coords and 'full_reconstructed' in coords['reconstruction_shapes']:
                expected_full_shape = coords['reconstruction_shapes']['full_reconstructed']
                current_shape = [final_img.data.shape[1], final_img.data.shape[2], final_img.data.shape[3]]
                if current_shape != expected_full_shape:
                    print(f"[WARNING] {patient_id} - Final dimensions {current_shape} don't match expected {expected_full_shape}")

        # Check if orientation needs adjustment
        print(f"[DEBUG] {patient_id} - Affine matrix:\n{final_img.affine}")

        output_path = out_dir / f"{patient_id}_reconstructed.nii.gz"
        final_img.save(output_path)
        print(f"[DEBUG] {patient_id} - Successfully saved to {output_path}")

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct full breast segmentations from cropped left/right parts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--coords_dir', type=str, required=True, help='Directory containing the coordinate files (e.g., .../preprocessed_crop-with-seg/data).')
    parser.add_argument('--segmentations_dir', type=str, required=True, help='Directory containing the segmentation files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the reconstructed segmentations.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers.')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of patients to process (set to 0 for no limit).')
    args = parser.parse_args()

    coords_dir = Path(args.coords_dir)
    seg_dir = Path(args.segmentations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Integrity Check ---
    available_seg_ids = {p.name.split('_')[0] for p in seg_dir.glob('*_left_segmentation.nii.gz')}
    print(f"Found {len(available_seg_ids)} unique patient IDs in the segmentation directory.")

    available_coord_folders = {path.parent.name.rsplit('_', 1)[0] for path in coords_dir.glob('*_left/cropping_coordinates.json')}
    print(f"Found {len(available_coord_folders)} unique patient IDs in the coordinates directory.")

    numeric_id_to_patient_id = {pid.split('_')[-1].lstrip('0'): pid for pid in available_coord_folders}
    coord_numeric_ids = set(numeric_id_to_patient_id.keys())

    missing_coords_ids = available_seg_ids - coord_numeric_ids
    missing_segs_ids = coord_numeric_ids - available_seg_ids

    if missing_coords_ids:
        print(f"\nWARNING: {len(missing_coords_ids)} patients have segmentations but are MISSING coordinate files.")
        print("IDs: ", sorted([int(i) for i in missing_coords_ids]))

    if missing_segs_ids:
        print(f"\nWARNING: {len(missing_segs_ids)} patients have coordinate files but are MISSING segmentations.")
        print("Full IDs: ", sorted([numeric_id_to_patient_id[i] for i in missing_segs_ids]))
    
    valid_numeric_ids = available_seg_ids.intersection(coord_numeric_ids)
    patient_ids = sorted([numeric_id_to_patient_id[nid] for nid in valid_numeric_ids])

    if not patient_ids:
        print("\nNo patients found with all required files. Halting reconstruction.")
        return

    print(f"\nFound {len(patient_ids)} patients with all required files to reconstruct.")
    if args.limit > 0:
        patient_ids = patient_ids[:args.limit]
        print(f"Processing a limit of {len(patient_ids)} patients.")

    reconstruct_fn = functools.partial(reconstruct_patient, 
                                       coords_dir=coords_dir, 
                                       seg_dir=seg_dir,
                                       out_dir=out_dir)
    
    if args.num_workers > 1 and len(patient_ids) > 1:
        print(f"\nStarting reconstruction with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            list(tqdm(pool.imap_unordered(reconstruct_fn, patient_ids), total=len(patient_ids)))
    else:
        print("\nStarting reconstruction with a single worker...")
        for pid in tqdm(patient_ids):
            reconstruct_fn(pid)

    print("\nReconstruction complete.")

if __name__ == "__main__":
    # Add information about TorchIO dimension handling
    print("\n[INFO] TorchIO dimension handling:")
    print("  - TorchIO loads as (C, D, H, W) by default")
    print("  - We permute to (C, W, H, D) for this script's logic")
    print("  - Concatenation is along width (dim=1)")
    print("  - Padding format is (left, right, bottom, top, back, front)\n")
    main()