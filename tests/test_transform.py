import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
import torchio as tio

def visualize_raw_and_transformed_image():
    """Load and visualize both raw and transformed images from the DUKE dataset"""
    
    # Create dataset with minimal transforms (just flip for orientation)
    minimal_transform = tio.Compose([
        tio.Flip(1),  # Just for viewing orientation
        tio.Lambda(lambda x: x)  # Identity transform
    ])
    
    dataset_raw = DUKE_Dataset3D(
        split='train',
        fold=0,
        fraction=0.1,
        transform=minimal_transform,
        image_crop=None,  # No cropping
        to_tensor=False  # Keep as torchio subject
    )
    
    # Create dataset with default transforms
    dataset_transformed = DUKE_Dataset3D(
        split='train',
        fold=0,
        fraction=0.1,
        transform=None,  # Use default transforms from dataset_3d_duke.py
        image_crop=(224, 224, 32),
        random_center=False,
        flip=False,  # Disable random flip for consistent comparison
        random_rotate=False,  # Disable random rotation for consistent comparison
        noise=False  # Disable noise for cleaner comparison
    )
    
    if len(dataset_raw) == 0:
        print("No data found in the dataset.")
        return
    
    # Get the same sample from both datasets
    sample_raw = dataset_raw[0]
    sample_transformed = dataset_transformed[0]
    
    uid = sample_raw['uid']
    target = sample_raw['target']
    
    print(f"Sample UID: {uid}")
    print(f"Target (Malignant): {target}")
    
    # Extract image data
    img_raw = sample_raw['source']
    img_transformed = sample_transformed['source']
    
    print(f"Raw image shape: {img_raw.data.shape}")
    print(f"Transformed image shape: {img_transformed.shape}")
    
    # Convert to numpy for visualization
    if hasattr(img_raw, 'data'):
        img_raw_np = img_raw.data.squeeze().numpy()
    else:
        img_raw_np = img_raw.squeeze().numpy()
    
    if isinstance(img_transformed, torch.Tensor):
        img_transformed_np = img_transformed.squeeze().numpy()
    else:
        img_transformed_np = img_transformed.data.squeeze().numpy()
    
    # Select middle slices for visualization
    if len(img_raw_np.shape) == 3:
        mid_slice_raw = img_raw_np.shape[2] // 2
        slice_raw = img_raw_np[:, :, mid_slice_raw]
        print(f"Raw: Showing middle slice {mid_slice_raw} of {img_raw_np.shape[2]} slices")
    else:
        slice_raw = img_raw_np
    
    if len(img_transformed_np.shape) == 3:
        mid_slice_trans = img_transformed_np.shape[2] // 2
        slice_trans = img_transformed_np[:, :, mid_slice_trans]
        print(f"Transformed: Showing middle slice {mid_slice_trans} of {img_transformed_np.shape[2]} slices")
    else:
        slice_trans = img_transformed_np
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Raw vs Transformed Image Comparison - UID: {uid}', fontsize=16)
    
    # Raw image
    im1 = axes[0, 0].imshow(slice_raw, cmap='gray')
    axes[0, 0].set_title(f'Raw Image\nShape: {img_raw_np.shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Transformed image
    im2 = axes[0, 1].imshow(slice_trans, cmap='gray')
    axes[0, 1].set_title(f'Transformed Image\nShape: {img_transformed_np.shape}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Histograms
    axes[1, 0].hist(img_raw_np.flatten(), bins=50, alpha=0.7, label='Raw')
    axes[1, 0].set_title('Raw Image Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(img_transformed_np.flatten(), bins=50, alpha=0.7, label='Transformed', color='orange')
    axes[1, 1].set_title('Transformed Image Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path('transform_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path.absolute()}")
    
    # Print statistics
    print("\n=== Image Statistics ===")
    print(f"Raw - Min: {img_raw_np.min():.4f}, Max: {img_raw_np.max():.4f}, Mean: {img_raw_np.mean():.4f}, Std: {img_raw_np.std():.4f}")
    print(f"Transformed - Min: {img_transformed_np.min():.4f}, Max: {img_transformed_np.max():.4f}, Mean: {img_transformed_np.mean():.4f}, Std: {img_transformed_np.std():.4f}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    try:
        visualize_raw_and_transformed_image()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()