import numpy as np
import SimpleITK as sitk


def n4bc(img_np, mask_np, num_iters=50, num_fitting_levels=5):
    """
    N4 Bias Field Correction
    
    Args:
        img_np: Input image as numpy array
        mask_np: Mask as numpy array
        num_iters: Number of iterations
        num_fitting_levels: Number of fitting levels
    
    Returns:
        Corrected image as numpy array
    """
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # Using N4BC parameters from Schwarzhans et. al (2024)
    corrector.SetMaximumNumberOfIterations([num_iters] * num_fitting_levels)

    img_sitk = sitk.GetImageFromArray(img_np)
    mask_sitk = sitk.GetImageFromArray(mask_np)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)

    corrected = corrector.Execute(img_sitk, mask_sitk)

    return sitk.GetArrayFromImage(corrected)


def plhe(img_np, num_segments=10):
    """
    Piecewise Linear Histogram Equalization
    
    Args:
        img_np: Input image as numpy array
        num_segments: Number of segments for histogram equalization
    
    Returns:
        Equalized image as numpy array
    """
    flat_img = img_np.astype(np.float32).flatten()

    sorted_vals = np.sort(flat_img)
    n_voxels = len(sorted_vals)
    segment_size = n_voxels // num_segments

    # Compute equal voxel count bins
    bins = [sorted_vals[i * segment_size] for i in range(num_segments)] + [sorted_vals[-1]]
    bins = np.unique(bins)

    corrected_img = np.zeros_like(img_np, dtype=np.float32)

    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        mask = (img_np >= lower) & (img_np <= upper)
        segment_voxels = img_np[mask]

        if segment_voxels.size == 0 or upper == lower:
            continue # skip invalid

        sorted_segment = np.sort(segment_voxels)
        ranks = np.searchsorted(sorted_segment, segment_voxels)
        cdf = ranks.astype(np.float32) / len(segment_voxels)

        # Linear stretch of voxel intensities within segment
        stretched = lower + cdf * (upper - lower)

        corrected_img[mask] = stretched

    return corrected_img
