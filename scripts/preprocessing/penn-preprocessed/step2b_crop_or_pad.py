from pathlib import Path 
import torchio as tio 
import torch
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm
import functools
from scipy.ndimage import label, zoom

# Final cropped/padded volume shape written to disk as (H, W, D).
# Bump D to enable a larger maximum slice count downstream; runtime crops
# in the dataset can still subselect any D' <= D.
TARGET_SHAPE = (256, 256, 32)


def _vertical_crop_bounds(
    bbox_top: int,
    bbox_bottom: int,
    image_height: int,
    target_height: int,
    *,
    top_pad_fraction: float = 0.15,
) -> tuple[int, int]:
    """Return (top_crop, bottom_crop) for tio.Crop on the H axis (dim 2).

    Fits the vertical mask/foreground bbox inside the crop window. Extra
    vertical space is biased above the ROI (small top pad) so the superior
    breast is not clipped while inferior background is trimmed first.
    """
    if image_height <= target_height:
        return 0, 0

    bbox_top = int(bbox_top)
    bbox_bottom = int(bbox_bottom)
    bbox_h = bbox_bottom - bbox_top + 1

    if bbox_h >= target_height:
        crop_start = (bbox_top + bbox_bottom) // 2 - target_height // 2
    else:
        extra = target_height - bbox_h
        top_pad = min(int(extra * top_pad_fraction), extra, max(4, extra // 8))
        crop_start = bbox_top - top_pad

    crop_start = max(0, min(crop_start, image_height - target_height))
    crop_end = crop_start + target_height

    # Keep the full bbox inside the window when possible.
    if bbox_top < crop_start:
        crop_start = bbox_top
    if bbox_bottom >= crop_end:
        crop_start = bbox_bottom - target_height + 1
    crop_start = max(0, min(crop_start, image_height - target_height))

    top_crop = crop_start
    bottom_crop = image_height - crop_start - target_height
    return top_crop, bottom_crop


def get_breast_crop_transform_from_mask(mask_3d: np.ndarray, image_height: int, target_height: int = 256):
    """Vertical crop from a binary mask bbox (preferred for new_penn step2b)."""
    mask_3d = np.asarray(mask_3d)
    if mask_3d.ndim != 3:
        raise ValueError(f"mask_3d must be 3D (W, H, D), got shape {mask_3d.shape}")

    proj = (mask_3d > 0).sum(axis=(0, 2))
    h_idx = np.where(proj > 0)[0]
    if len(h_idx) == 0:
        crop_start = (image_height - target_height) // 2
        top_crop = max(0, crop_start)
        bottom_crop = max(0, image_height - target_height - top_crop)
        return tio.Crop((0, 0, top_crop, bottom_crop, 0, 0))

    top_crop, bottom_crop = _vertical_crop_bounds(
        int(h_idx[0]), int(h_idx[-1]), image_height, target_height
    )
    return tio.Crop((0, 0, top_crop, bottom_crop, 0, 0))


def get_breast_crop_transform(image, target_height=256, downsample_factor=4):
    """Calculates a robust crop transform by localizing the largest connected component (breast tissue)."""
    image_height = image.shape[2]
    # Use a downsampled image for efficient processing
    with torch.no_grad():
        original_data = image.data[0].numpy()
        downsampled_data = zoom(original_data, 1 / downsample_factor, order=1)

    # Identify foreground using a quantile threshold
    threshold = np.quantile(downsampled_data, 0.90)
    foreground = downsampled_data > threshold
    
    # Find the largest connected component to isolate the breast
    labels, num_features = label(foreground)
    if num_features > 0:
        largest_component_label = np.argmax([np.sum(labels == i) for i in range(1, num_features + 1)]) + 1
        breast_mask = labels == largest_component_label
        
        fg_indices = np.argwhere(breast_mask.sum(axis=(0, 2)) > 0)
        if len(fg_indices) > 0:
            scale = downsample_factor
            bbox_top = int(fg_indices.min() * scale)
            bbox_bottom = int(fg_indices.max() * scale)
            top_crop, bottom_crop = _vertical_crop_bounds(
                bbox_top, bbox_bottom, image_height, target_height
            )
            return tio.Crop((0, 0, top_crop, bottom_crop, 0, 0))

    # Fallback: center crop
    if image_height <= target_height:
        return tio.Crop((0, 0, 0, 0, 0, 0))
    crop_start = (image_height - target_height) // 2
    return tio.Crop((0, 0, crop_start, image_height - crop_start - target_height, 0, 0))


def preprocess(path_dir, path_root_in_data_str, path_root_out_data_str):
    path_root_in_data = Path(path_root_in_data_str)
    path_root_out_data = Path(path_root_out_data_str)

    # --- Check if already processed ---
    path_out_left = path_root_out_data / f"{path_dir.relative_to(path_root_in_data)}_left"
    path_out_right = path_root_out_data / f"{path_dir.relative_to(path_root_in_data)}_right"
    if path_out_left.exists() and path_out_right.exists():
        # print(f"Skipping {path_dir.name} as output directories already exist.")
        return

    try:
        # --- Load and resample the reference 'pre' image ---
        pre_img_orig = tio.ScalarImage(path_dir / 'pre.nii.gz')
        pre_img_orig.data = torch.rot90(pre_img_orig.data, k=2, dims=(0, 1))
        # --- Define transforms ---
        # Resample to a standard orientation and spacing
        resample_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample((1.0, 1.0, 1.0)),
        ])
        pre_img_resampled = resample_transform(pre_img_orig)

        # --- Split left and right sides ---
        width = pre_img_resampled.shape[1]
        split_transforms = {
            'right': tio.Crop((width // 2, 0, 0, 0, 0, 0)),
            'left': tio.Crop((0, width // 2, 0, 0, 0, 0)),
        }

        # --- Get cropping transforms for each side (run once on 'pre' image) ---
        crop_transforms = {}
        for side in ['left', 'right']:
            pre_img_side = split_transforms[side](pre_img_resampled)
            crop_transforms[side] = get_breast_crop_transform(pre_img_side, target_height=256)

        # --- Process all images in the directory ---
        for path_img in path_dir.glob('*.nii.gz'):
            img_orig = tio.ScalarImage(path_img)
            if img_orig.shape[1] < 100:
                print(f"Skipping {path_img.name} due to small x-dimension: {img_orig.shape[1]}")
                continue
            
            img_resampled = resample_transform(img_orig)

            for side in ['left', 'right']:
                path_out_dir = path_root_out_data / f"{path_dir.relative_to(path_root_in_data)}_{side}"
                path_out_dir.mkdir(exist_ok=True, parents=True)

                # Apply transforms
                final_transform = tio.Compose([
                    split_transforms[side],
                    crop_transforms[side],
                    tio.CropOrPad(TARGET_SHAPE, padding_mode=0),
                ])
                img_final = final_transform(img_resampled)
                img_final.save(path_out_dir / path_img.name)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Skipping {path_dir.name} due to error: {e}")

if __name__ == "__main__":
    path_root = Path(r'D:\PENN-MRI')
    path_root_in_data = path_root / 'penn-preprocessed2' / 'data'
    path_root_out = path_root / 'penn-preprocessed_cropped2'
    path_root_out_data = path_root_out / 'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    path_patients = [p for p in path_root_in_data.iterdir() if p.is_dir()]
    
    # Option 1: Multi-CPU 
    # Convert Path objects to strings for serialization
    path_root_in_data_str = str(path_root_in_data)
    path_root_out_data_str = str(path_root_out_data)
    
    partial_preprocess = functools.partial(preprocess, 
                                           path_root_in_data_str=path_root_in_data_str, 
                                           path_root_out_data_str=path_root_out_data_str)
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial_preprocess, path_patients), total=len(path_patients)):
            pass

    # Option 2: Single-CPU 
    # for path_dir in tqdm(path_patients):
    #     preprocess(path_dir)    