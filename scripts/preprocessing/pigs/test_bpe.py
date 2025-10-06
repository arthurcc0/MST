
import numpy as np
import matplotlib.pyplot as plt

def examine_fgt_channels(fgt_path):
    """
    Examine each channel of the FGT mask to understand the data structure
    """
    print("Loading FGT data...")
    fgt_data = np.load(fgt_path)
    print(f"Original FGT shape: {fgt_data.shape}")
    print(f"Original FGT dtype: {fgt_data.dtype}")
    
    # Examine each channel
    for channel in range(fgt_data.shape[0]):
        print(f"\n=== CHANNEL {channel} ===")
        channel_data = fgt_data[channel]
        print(f"Channel {channel} shape: {channel_data.shape}")
        print(f"Channel {channel} dtype: {channel_data.dtype}")
        print(f"Channel {channel} range: [{channel_data.min():.6f}, {channel_data.max():.6f}]")
        print(f"Channel {channel} mean: {channel_data.mean():.6f}")
        
        # Count unique values (sample if too many)
        unique_vals = np.unique(channel_data)
        print(f"Channel {channel} unique values count: {len(unique_vals)}")
        
        if len(unique_vals) <= 10:
            print(f"Channel {channel} unique values: {unique_vals}")
        else:
            print(f"Channel {channel} first 10 unique values: {unique_vals[:10]}")
            print(f"Channel {channel} last 10 unique values: {unique_vals[-10:]}")
        
        # Check for binary-like data
        binary_like = np.all((channel_data >= 0) & (channel_data <= 1))
        print(f"Channel {channel} is binary-like (0-1): {binary_like}")
        
        # Count non-zero voxels
        non_zero_count = np.sum(channel_data > 0)
        total_voxels = channel_data.size
        non_zero_percentage = 100 * non_zero_count / total_voxels
        print(f"Channel {channel} non-zero voxels: {non_zero_count}/{total_voxels} ({non_zero_percentage:.1f}%)")
        
        # Check for high-confidence voxels (>0.5 if probabilities)
        if binary_like:
            high_conf_count = np.sum(channel_data > 0.5)
            high_conf_percentage = 100 * high_conf_count / total_voxels
            print(f"Channel {channel} high-confidence voxels (>0.5): {high_conf_count}/{total_voxels} ({high_conf_percentage:.1f}%)")
    
    # Compare channels
    print(f"\n=== CHANNEL COMPARISON ===")
    for i in range(fgt_data.shape[0]):
        for j in range(i+1, fgt_data.shape[0]):
            correlation = np.corrcoef(fgt_data[i].flatten(), fgt_data[j].flatten())[0, 1]
            print(f"Correlation between channel {i} and {j}: {correlation:.4f}")
    
    # Visualize middle slice of each channel
    print(f"\n=== VISUALIZATION ===")
    middle_slice_idx = fgt_data.shape[-1] // 2  # Middle slice in z-direction
    
    fig, axes = plt.subplots(1, fgt_data.shape[0], figsize=(15, 5))
    if fgt_data.shape[0] == 1:
        axes = [axes]
    
    for channel in range(fgt_data.shape[0]):
        channel_slice = fgt_data[channel, :, :, middle_slice_idx+4]
        im = axes[channel].imshow(channel_slice, cmap='gray')
        axes[channel].set_title(f'Channel {channel}\nSlice {middle_slice_idx}')
        axes[channel].axis('off')
        plt.colorbar(im, ax=axes[channel], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('fgt_channels_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fgt_data

if __name__ == "__main__":
    # Path to your FGT mask
    fgt_path = r"D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\duke_output\fgt\Breast_MRI_001.npy"
    
    fgt_data = examine_fgt_channels(fgt_path)
    
    print(f"\n=== SUMMARY ===")
    print(f"FGT data shape: {fgt_data.shape}")
    print(f"Number of channels: {fgt_data.shape[0]}")
    print(f"Spatial dimensions: {fgt_data.shape[1:]} (H x W x D)")