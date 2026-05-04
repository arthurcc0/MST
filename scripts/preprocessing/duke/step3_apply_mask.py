from pathlib import Path 
import SimpleITK as sitk 
from tqdm import tqdm 
import numpy as np
from multiprocessing import Pool, cpu_count
import os

PATH_ROOT = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed_crop_n4bc_plhe_fast_full-a\data')

def subtraction(pre, post):
    pre_img = sitk.GetArrayFromImage(sitk.ReadImage(str(pre)))
    post_img = sitk.GetArrayFromImage(sitk.ReadImage(str(post)))
    subtraction = post_img - pre_img
    subtraction = subtraction - subtraction.min()
    return subtraction

def process_single_mask(mask_path):
    """Process a single mask - designed for multiprocessing"""
    try:
        mask = Path(mask_path)
        patient_dir = mask.parent
        pre = patient_dir / 'pre.nii.gz'
        post = patient_dir / 'post_1.nii.gz'
        
        # Check if required files exist
        if not pre.exists() or not post.exists():
            return f"Skipped {patient_dir.name}: Missing pre or post files"
        
        sub = subtraction(pre, post)
        sub_masked = sub * sitk.GetArrayFromImage(sitk.ReadImage(str(mask)))
        sitk.WriteImage(sitk.GetImageFromArray(sub_masked), str(patient_dir / 'sub.nii.gz'))
        
        return f"Processed {patient_dir.name}"
    except Exception as e:
        return f"Error processing {mask_path}: {str(e)}"

def apply_mask_to_patient(num_processes=None):
    """Apply masks to patients using multiprocessing"""
    
    # Get all mask paths
    masks = list(PATH_ROOT.glob('*/mask.nii.gz'))
    
    if not masks:
        print("No mask files found!")
        return
    
    # Use all available CPUs if not specified
    if num_processes is None:
        num_processes = min(cpu_count(), len(masks))
    
    print(f"Processing {len(masks)} masks using {num_processes} processes...")
    
    # Convert Path objects to strings for multiprocessing
    mask_paths = [str(mask) for mask in masks]
    
    # Process masks in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_mask, mask_paths),
            total=len(mask_paths),
            desc="Processing masks"
        ))
    
    # Print results summary
    successful = sum(1 for r in results if r.startswith("Processed"))
    errors = sum(1 for r in results if r.startswith("Error"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    
    # Print any errors or skipped items
    for result in results:
        if result.startswith("Error") or result.startswith("Skipped"):
            print(f"  {result}")
    
    print("Done!")

if __name__ == '__main__':
    apply_mask_to_patient()