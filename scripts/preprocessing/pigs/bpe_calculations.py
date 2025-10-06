import numpy as np

def calculate_bpe_mask(pre_img, post_img, fgt_mask, enhancement_threshold=1.0):
    """
    Calculate BPE mask with proper shape handling and realistic enhancement values
    """
    # Ensure all inputs have same shape
    assert pre_img.shape == post_img.shape == fgt_mask.shape, \
        f"Shape mismatch: Pre={pre_img.shape}, Post={post_img.shape}, Mask={fgt_mask.shape}"
    
    fgt_indices = fgt_mask > 1e-6
    fgt_mask[fgt_indices] = 1
    enhancement = np.zeros_like(post_img, dtype=np.float32)
    
    if np.any(fgt_indices):
        pre_fgt = pre_img[fgt_indices]
        post_fgt = post_img[fgt_indices]
        
        valid_pre = pre_fgt > 1.0 
        
        if np.any(valid_pre):
            epsilon = 1e-6
            pre_valid = pre_fgt[valid_pre]
            post_valid = post_fgt[valid_pre]
            
            enhancement_valid = (post_valid - pre_valid) / (pre_valid + epsilon) * 100.0
            
         
            fgt_coords = np.where(fgt_indices)
            valid_coords = tuple(coord[valid_pre] for coord in fgt_coords)
            enhancement[valid_coords] = enhancement_valid
    
    bpe_mask = (fgt_mask > 0) & (enhancement > enhancement_threshold)
   # bpe_mask = ((fgt_mask)*post_img)
    print("="*12)
    print(bpe_mask.shape)
    print(bpe_mask.max())
    print(bpe_mask.min())
    print(bpe_mask)
    print("="*12)
    return bpe_mask.astype(np.uint16)
def calculate_relative_enhancement(pre_img, post_img, mask):
    """
    Calculates mean and median relative (percent) enhancement in the fibroglandular mask.
    RE = ((SI_post - SI_pre) / SI_pre) * 100
    """
    pre_vals = pre_img[mask > 0]
    post_vals = post_img[mask > 0]

    epsilon = 1e-6
    re_vals = (post_vals - pre_vals) / (pre_vals + epsilon) * 100.0

    mean_re = np.mean(re_vals)
    median_re = np.median(re_vals)
    std_re = np.std(re_vals)
    
    return mean_re, median_re, std_re

def calculate_volumetric_bpe(pre_img, post_img, mask, voxel_spacing=(0,0,0), enhancement_threshold=20.0):  # voxel (x,y,z)
    """
    Calculates:
      - BPE Volume (in cm³)
      - BPE Fraction (fraction of FGT above threshold)
    
    threshold is percent enhancement (e.g. 50%).
    voxel_spacing = (row_spacing, col_spacing, slice_thickness) in mm.
    """
    pre_vals = pre_img[mask > 0]
    post_vals = post_img[mask > 0]

    epsilon = 1e-6
    re_vals = (post_vals - pre_vals) / (pre_vals + epsilon) * 100.0

    bpe_voxels = np.sum(re_vals > enhancement_threshold)
    
    bpe_mask = np.zeros(post_img.shape, dtype=bool)
    mask_coords = np.where(mask > 0)
    enhanced_coords = tuple(coord[re_vals > enhancement_threshold] for coord in mask_coords)
    if len(enhanced_coords[0]) > 0:
        bpe_mask[enhanced_coords] = True

    total_fgt_voxels = len(pre_vals)

    if total_fgt_voxels == 0:
        bpe_fraction = 0.0
    else:
        bpe_fraction = bpe_voxels / total_fgt_voxels
        
    # Compute voxel volume in mm³ if spacing is known
    row_spacing, col_spacing, slice_thickness = voxel_spacing
    voxel_volume_cm3 = (row_spacing * col_spacing * slice_thickness) / 1000
    bpe_volume_cm3 = bpe_voxels * voxel_volume_cm3
    
    return bpe_volume_cm3, bpe_fraction, bpe_mask

def preprocess_fgt_mask(fgt_path):
    """
    Preprocess FGT mask from multi-channel probabilities to single segmentation
    
    Args:
        fgt_path: Path to FGT mask file
        
    Returns:
        fgt_segmentation: Single-channel segmentation mask (160, 448, 448)
        fgt_original: Original multi-channel data for reference
    """
    # Load original FGT data
    fgt_original = np.load(fgt_path)
    print(f"Original FGT shape: {fgt_original.shape}")
    print(f"Original FGT dtype: {fgt_original.dtype}")
    
    fgt_segmentation = np.argmax(fgt_original, axis=0)
    fgt_segmentation = (fgt_segmentation == 2).astype(np.uint8)
    
    fgt_segmentation = np.transpose(fgt_segmentation, (2, 0, 1))
    
    print(f"FGT after selecting channel 1 and transpose: {fgt_segmentation.shape}")
    print(f"FGT unique values: {np.unique(fgt_segmentation)}")
    
    unique_values = np.unique(fgt_segmentation)
    if len(unique_values) > 20:
        print(f"  Too many unique values ({len(unique_values)}), showing summary:")
        print(f"  Min value: {unique_values[0]}")
        print(f"  Max value: {unique_values[-1]}")
        print(f"  Non-zero voxels: {np.sum(fgt_segmentation > 0)}/{fgt_segmentation.size}")
    else:
        for value in unique_values:
            count = np.sum(fgt_segmentation == value)
            percentage = 100 * count / fgt_segmentation.size
            print(f"  Value {value}: {count} voxels ({percentage:.1f}%)")
    
    return fgt_segmentation, fgt_original

def get_slices_check(volume):
    """Extract slices at 75%, 50%, and 25% depth"""
    depth = volume.shape[2]
    slices = np.array([
        volume[:, :, int(depth * 0.75)],
        volume[:, :, depth // 2],
        volume[:, :, int(depth * 0.25)]
    ])
    return slices

def analyze_fgt_slices(fgt_data):
    """
    Analyze FGT slices using get_slices_check
    
    Args:
        fgt_data: Preprocessed FGT segmentation data (160, 448, 448)
        
    Returns:
        slices_fgt: Array of key slices (3, 448, 448)
        slice_stats: Dictionary with statistics for each slice
    """
    # Apply get_slices_check to analyze key slices
    slices_fgt = get_slices_check(fgt_data)
    
    print(f"Extracted FGT slices shape: {slices_fgt.shape}")
    
    # Analyze each slice
    slice_positions = ['75%', '50%', '25%']
    depth = fgt_data.shape[2]
    slice_indices = [int(depth * 0.75), depth // 2, int(depth * 0.25)]
    
    slice_stats = {}
    
    for i, (pos, idx) in enumerate(zip(slice_positions, slice_indices)):
        slice_data = slices_fgt[i]
        print(f"\nFGT Slice {i+1} ({pos} depth, z={idx}):")
        print(f"  Shape: {slice_data.shape}")
        print(f"  Unique labels: {np.unique(slice_data)}")
        
        # Count pixels for each label
        label_stats = {}
        for label in np.unique(slice_data):
            count = np.sum(slice_data == label)
            percentage = 100 * count / slice_data.size
            label_stats[label] = {'count': count, 'percentage': percentage}
            print(f"  Label {label}: {count} pixels ({percentage:.1f}%)")
        
        slice_stats[pos] = {
            'index': idx,
            'shape': slice_data.shape,
            'labels': label_stats
        }
    
    return slices_fgt, slice_stats

def preprocess_breast_mask(breast_mask_path):
    """
    Preprocess breast mask to match image orientation
    
    Args:
        breast_mask_path: Path to breast mask file
        
    Returns:
        breast_mask: Oriented breast mask (160, 448, 448)
    """
    # Load breast mask
    breast_mask = np.load(breast_mask_path)
    print(f"Original breast mask shape: {breast_mask.shape}")
    
    # Handle different input formats
    if len(breast_mask.shape) == 4:
        # Multi-channel, use first channel
        breast_mask = breast_mask[0]
        print(f"Using first channel, shape: {breast_mask.shape}")
    
    # If shape is (448, 448, 160), transpose to (160, 448, 448)
    if breast_mask.shape == (448, 448, 160):
        breast_mask = np.transpose(breast_mask, (2, 0, 1))
        print(f"Transposed breast mask to: {breast_mask.shape}")
    
    return breast_mask
