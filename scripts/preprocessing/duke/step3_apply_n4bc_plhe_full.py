from pathlib import Path
import numpy as np
import torchio as tio
import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import tqdm
import functools
import sys
import os

# Add the parent directory to the path to import n4bc and plhe functions
sys.path.append(str(Path(__file__).parent))
from n4bc_plhe_fast import n4bc, plhe


def load_breast_mask_nifti(mask_path):
    """Load breast mask from NIfTI file and convert to numpy array"""
    if mask_path.exists():
        mask_img = tio.ScalarImage(mask_path)
        return mask_img.data.numpy()[0]  # Remove channel dimension
    else:
        print(f"Warning: Mask not found at {mask_path}")
        return None


def process_patient_directory(patient_dir, input_root, output_root, mask_root, apply_n4bc=True, apply_plhe=True):
    """
    Process cropped pre and post_1 images: apply N4BC/PLHE, calculate subtraction, apply masks
    
    Args:
        patient_dir: Path to patient directory containing pre.nii.gz and post_1.nii.gz
        input_root: Root path of input data
        output_root: Root path of output data
        mask_root: Root path of cropped masks
        apply_n4bc: Whether to apply N4 Bias Field Correction
        apply_plhe: Whether to apply Piecewise Linear Histogram Equalization
    """
    try:
        input_root = Path(input_root)
        output_root = Path(output_root)
        mask_root = Path(mask_root)
        
        # Look for pre and post_1 images
        pre_file = patient_dir / 'pre.nii.gz'
        post_file = patient_dir / 'post_1.nii.gz'
        
        if not pre_file.exists():
            print(f"Pre image not found in {patient_dir}")
            return
            
        if not post_file.exists():
            print(f"Post_1 image not found in {patient_dir}")
            return
        
        # Load pre and post images
        pre_img = tio.ScalarImage(pre_file)
        post_img = tio.ScalarImage(post_file)
        
        pre_np = pre_img.data.numpy()[0]  # Remove channel dimension
        post_np = post_img.data.numpy()[0]  # Remove channel dimension
        
        # Determine side (left or right) from patient directory name
        patient_name = patient_dir.name
        if '_left' in patient_name:
            side = 'left'
            base_name = patient_name.replace('_left', '')
        elif '_right' in patient_name:
            side = 'right'
            base_name = patient_name.replace('_right', '')
        else:
            print(f"Cannot determine side from directory name: {patient_name}")
            return
        
        # Find corresponding mask
        mask_dir = mask_root / f"{base_name}_{side}"
        mask_file = mask_dir / f"{base_name}.nii.gz"
        
        if not mask_file.exists():
            print(f"Mask not found at {mask_file}")
            return
        
        # Load mask
        mask_np = load_breast_mask_nifti(mask_file)
        if mask_np is None:
            return
        
        # Ensure all images have the same shape
        if pre_np.shape != post_np.shape:
            print(f"Shape mismatch between pre and post images in {patient_dir}")
            return
            
        if pre_np.shape != mask_np.shape:
            print(f"Resizing mask to match image shape for {patient_dir}")
            # Resize mask to match image
            mask_sitk = sitk.GetImageFromArray(mask_np)
            pre_sitk = sitk.GetImageFromArray(pre_np)
            mask_sitk = sitk.Resample(mask_sitk, pre_sitk)
            mask_np = sitk.GetArrayFromImage(mask_sitk)
        
        # Convert mask to binary (0 or 1)
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        processed_pre = pre_np.copy()
        processed_post = post_np.copy()
        
        # Apply N4 Bias Field Correction to both images if requested
        if apply_n4bc:
            print(f"Applying FAST N4BC to pre and post images in {patient_dir.name}")
            processed_pre = n4bc(processed_pre, mask_np)
            processed_post = n4bc(processed_post, mask_np)
        
        # Apply Piecewise Linear Histogram Equalization to both images if requested
        if apply_plhe:
            print(f"Applying PLHE to pre and post images in {patient_dir.name}")
            processed_pre = plhe(processed_pre)
            processed_post = plhe(processed_post)
        
        # Calculate subtraction (post_1 - pre)
        subtraction = processed_post - processed_pre
        
        # Apply mask to subtraction image
        masked_subtraction = subtraction * mask_np
        
        # Create output directory
        relative_path = patient_dir.relative_to(input_root)
        output_dir = output_root / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames with processing suffix
        suffix = []
        if apply_n4bc:
            suffix.append('n4bc_fast')
        if apply_plhe:
            suffix.append('plhe')
        
        suffix_str = '_' + '_'.join(suffix) if suffix else ''
        
        # Save processed images
        # Save processed pre
        processed_pre_tensor = processed_pre[np.newaxis, ...]  # Add channel dimension
        pre_output = tio.ScalarImage(tensor=processed_pre_tensor, affine=pre_img.affine)
        pre_output.save(output_dir / f"pre.nii.gz")
        
        # Save processed post
        processed_post_tensor = processed_post[np.newaxis, ...]  # Add channel dimension
        post_output = tio.ScalarImage(tensor=processed_post_tensor, affine=post_img.affine)
        post_output.save(output_dir / f"post_1.nii.gz")
        
        # Save masked subtraction
        subtraction_tensor = masked_subtraction[np.newaxis, ...]  # Add channel dimension
        sub_output = tio.ScalarImage(tensor=subtraction_tensor, affine=pre_img.affine)
        sub_output.save(output_dir / f"sub.nii.gz")
        
        # Also save the mask for reference
        mask_tensor = mask_np[np.newaxis, ...]  # Add channel dimension
        mask_output = tio.ScalarImage(tensor=mask_tensor, affine=pre_img.affine)
        mask_output.save(output_dir / "mask.nii.gz")
        
        print(f"Successfully processed: {patient_dir.name}")
        
    except Exception as e:
        print(f"Error processing {patient_dir.name}: {str(e)}")


if __name__ == "__main__":
    # FULL PROCESSING VERSION - Process ALL directories
    
    # Input directory containing cropped pre and post_1 images
    path_root_in = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed_crop-a\data')
    
    # Output directory for processed images
    path_root_out = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed_crop_n4bc_plhe_fast_full-a')
    path_root_out_data = path_root_out / 'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
    
    # Mask directory (from step2c_crop_masks.py)
    path_mask_root = Path(r'D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\cropped-masks\data')
    
    # Processing options
    APPLY_N4BC = True
    APPLY_PLHE = True
    
    print("="*80)
    print("FULL DATASET PROCESSING - ALL DIRECTORIES")
    print("="*80)
    print(f"Input: {path_root_in}")
    print(f"Output: {path_root_out_data}")
    print(f"Masks: {path_mask_root}")
    print("="*80)
    
    # Get all patient directories (should have names like "patient_001_left", "patient_001_right", etc.)
    if path_root_in.exists():
        patient_dirs = [d for d in path_root_in.iterdir() if d.is_dir()]
        print(f"Found {len(patient_dirs)} patient directories to process")
        
        # Filter to only left and right directories
        patient_dirs = [d for d in patient_dirs if ('_left' in d.name or '_right' in d.name)]
        print(f"Found {len(patient_dirs)} left/right patient directories to process")
        
        # NO LIMIT - PROCESS ALL DIRECTORIES
        print(f"FULL MODE: Processing ALL {len(patient_dirs)} directories")
        
    else:
        print(f"❌ Input directory does not exist: {path_root_in}")
        print("Please check the path to your cropped images")
        sys.exit(1)
    
    if not patient_dirs:
        print("❌ No patient directories found in input directory")
        print("Expected directory names like: 'Breast_MRI_001_left', 'Breast_MRI_001_right'")
        sys.exit(1)
    
    # Check if mask directory exists
    if not path_mask_root.exists():
        print(f"❌ Mask directory does not exist: {path_mask_root}")
        print("Please run step2c_crop_masks.py first to generate the cropped masks")
        sys.exit(1)
    
    # Estimate processing time
    estimated_time_minutes = len(patient_dirs) * 4.4 / 60  # Based on test run: ~4.4 seconds per directory
    print(f"⏱️  Estimated processing time: {estimated_time_minutes:.1f} minutes ({estimated_time_minutes/60:.1f} hours)")
    
    # Multi-CPU processing with 2 processes for stability
    partial_process = functools.partial(
        process_patient_directory,
        input_root=str(path_root_in),
        output_root=str(path_root_out_data),
        mask_root=str(path_mask_root),
        apply_n4bc=APPLY_N4BC,
        apply_plhe=APPLY_PLHE
    )
    
    print(f"\n🚀 Starting FULL processing with N4BC={'ON' if APPLY_N4BC else 'OFF'}, PLHE={'ON' if APPLY_PLHE else 'OFF'}")
    print(f"Using 2 CPU processes for stability")
    
    with Pool(processes=2) as pool:
        for _ in tqdm(pool.imap_unordered(partial_process, patient_dirs), total=len(patient_dirs)):
            pass

    print("\n✅ FULL N4BC and PLHE processing completed!")
    print(f"Results saved to: {path_root_out_data}")
