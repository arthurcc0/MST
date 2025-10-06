import numpy as np
import nibabel as nib
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk

def debug_fgt(fgt_path):
    fgt = np.load(fgt_path)
    print(fgt.shape)
    print(fgt.dtype)
    for i in range(fgt.shape[0]):
        print(f"\nChannel {i}:")
        print(f"  Range: [{fgt[i].min():.6f}, {fgt[i].max():.6f}]")
        print(f"  Mean: {fgt[i].mean():.6f}")
        print(f"  Std: {fgt[i].std():.6f}")
    fgt_argmax = np.argmax(fgt, axis=0)
    unique, counts = np.unique(fgt_argmax, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = 100 * count / fgt_argmax.size
        print(f"Label {label}: {count} voxels ({percentage:.2f}%)")
    binary_mask = (fgt_argmax == 2).astype(np.uint16)
    print(f"Binary mask shape: {binary_mask.shape}")
    print(f"Total FGT voxels: {np.sum(binary_mask)}")  # Should be 318566

    # Find which slices contain FGT
    fgt_per_slice = np.sum(binary_mask, axis=(0, 1))  # Sum over H and W
    slices_with_fgt = np.where(fgt_per_slice > 0)[0]
    print(f"Slices containing FGT: {slices_with_fgt}")
    print(f"Number of slices with FGT: {len(slices_with_fgt)}")

    # Visualize a slice with the most FGT
    best_slice = np.argmax(fgt_per_slice)
    print(f"Slice with most FGT: {best_slice} ({fgt_per_slice[best_slice]} voxels)")

    import matplotlib.pyplot as plt
    plt.imshow(binary_mask[:, :, best_slice], cmap='gray')
    plt.title(f'FGT Binary Mask - Slice {best_slice}')
    plt.show()
    print(binary_mask.shape)
    print(f"Channel 2 range: [{fgt_argmax.min():.6f}, {fgt_argmax.max():.6f}]")
    print(f"Channel 2 mean: {fgt_argmax.mean():.6f}")
    print(f"Channel 2 std: {fgt_argmax.std():.6f}")
    
    plt.figure(figsize=(12, 8))
    
    values = fgt_argmax.flatten()
    
    non_zero_values = values[values > 0]
    print(f"Total voxels: {len(values)}")
    print(f"Non-zero voxels: {len(non_zero_values)} ({100*len(non_zero_values)/len(values):.1f}%)")
    
    plt.subplot(2, 2, 1)
    plt.hist(values, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('FGT Channel 2 - All Values')
    plt.xlabel('Probability Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    if len(non_zero_values) > 0:
        plt.hist(non_zero_values, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.title('FGT Channel 2 - Non-Zero Values Only')
        plt.xlabel('Probability Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(values, bins=100, alpha=0.7, color='green', edgecolor='black', log=True)
    plt.title('FGT Channel 2 - Log Scale')
    plt.xlabel('Probability Value')
    plt.ylabel('Frequency (log)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    high_conf_values = values[values > 0.5]
    if len(high_conf_values) > 0:
        plt.hist(high_conf_values, bins=25, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'High Confidence Values (>0.5)\n{len(high_conf_values)} voxels')
        plt.xlabel('Probability Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No values > 0.5', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('High Confidence Values (>0.5)')
    
    plt.tight_layout()
    plt.savefig('fgt_channel2_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== VALUE STATISTICS ===")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(values, p)
        print(f"{p}th percentile: {val:.6f}")
    
    print(f"\n=== VALUE RANGES ===")
    ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in ranges:
        count = np.sum((values >= low) & (values < high))
        percentage = 100 * count / len(values)
        print(f"[{low:.1f}, {high:.1f}): {count} voxels ({percentage:.1f}%)")

    nii_img = sitk.GetImageFromArray(binary_mask)
    sitk.WriteImage(nii_img, "fgt_channel2.nii.gz")
     
if __name__ == "__main__":
    fgt_mask_path = r"D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\duke_output\fgt\Breast_MRI_001.npy"
    debug_fgt(fgt_mask_path)