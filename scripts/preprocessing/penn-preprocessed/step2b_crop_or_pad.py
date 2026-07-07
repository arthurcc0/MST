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

# Mask cleanup before vertical bbox (new_penn step2b).
MASK_CC_MIN_FRACTION = 0.05  # drop 3D components smaller than this × largest
MASK_CC_SUPERIOR_BRIGHT_ROWS = 32  # extend bbox top using pre intensity within this band
BRIGHT_QUANTILE = 0.90

# Vertical crop blend (v3 small-breast top pad + v2-like large-breast centering).
# When mask bbox height >= target crop height, centering on the mask midpoint
# shifts the window inferiorly (mask often extends below the breast). Instead
# anchor the bbox top slightly below the crop top and do not pull the window
# down to include the inferior mask edge.
# Fraction of target H placed above mask top in large-breast / blend-superior modes.
#   > 0 : extra background above the breast (crop shifts up).
#   = 0 : mask top on crop top.
#   < 0 : crop shifts down (trim a little superior tissue, show more inferior field).
# Try large breasts: 0.0 (default), then -0.03 .. -0.08 if bottom feels clipped.
LARGE_BBOX_SUPERIOR_ANCHOR = -0.05
LARGE_BBOX_BLEND_START_FRAC = 0.80  # begin blending toward superior anchor above this × target H
TOP_PAD_FRACTION_SMALL = 0.15  # extra space above bbox when bbox fits inside crop


def _keep_largest_3d_component(mask_3d: np.ndarray, min_fraction: float = MASK_CC_MIN_FRACTION) -> np.ndarray:
    """Keep only the largest 3D connected component; ignore small island artifacts."""
    binary = np.asarray(mask_3d) > 0
    if not binary.any():
        return binary.astype(np.uint8)

    labels, num_features = label(binary)
    if num_features == 0:
        return binary.astype(np.uint8)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_size = int(sizes.max())
    if largest_size == 0:
        return binary.astype(np.uint8)

    min_size = max(1, int(largest_size * min_fraction)) if min_fraction > 0 else 1
    candidates = [lbl for lbl in range(1, num_features + 1) if sizes[lbl] >= min_size]
    if not candidates:
        return binary.astype(np.uint8)

    largest_label = max(candidates, key=lambda lbl: int(sizes[lbl]))
    return (labels == largest_label).astype(np.uint8)


def _vertical_h_extent(
    volume_3d: np.ndarray,
    *,
    image_3d: np.ndarray | None = None,
    bright_quantile: float = BRIGHT_QUANTILE,
    superior_bright_rows: int = MASK_CC_SUPERIOR_BRIGHT_ROWS,
) -> tuple[int, int] | None:
    """Row indices (H axis) spanning foreground; optionally extend superior edge with pre."""
    vol = np.asarray(volume_3d) > 0
    if not vol.any():
        return None

    proj = vol.sum(axis=(0, 2)) > 0
    h_idx = np.where(proj)[0]
    top, bottom = int(h_idx[0]), int(h_idx[-1])

    bbox_h = bottom - top + 1
    if image_3d is not None:
        img = np.asarray(image_3d, dtype=np.float32)
        positive = img[img > 0]
        thr = float(np.quantile(positive if positive.size else img, bright_quantile))
        bright_rows = np.where((img > thr).sum(axis=(0, 2)) > 0)[0]
        if bright_rows.size:
            # Pull top upward; use a wider band on large breasts.
            band = superior_bright_rows
            img_h = int(img.shape[2])  # TorchIO (W, H, D)
            if bbox_h >= int(0.5 * img_h):
                band = max(band, int(0.12 * bbox_h), 48)
            near_superior = bright_rows[bright_rows <= top + band]
            if near_superior.size:
                top = min(top, int(near_superior.min()))

    return top, bottom


def _vertical_crop_bounds(
    bbox_top: int,
    bbox_bottom: int,
    image_height: int,
    target_height: int,
    *,
    top_pad_fraction: float = TOP_PAD_FRACTION_SMALL,
    large_bbox_superior_anchor: float = LARGE_BBOX_SUPERIOR_ANCHOR,
    large_bbox_blend_start_frac: float = LARGE_BBOX_BLEND_START_FRAC,
) -> tuple[int, int, str]:
    """Return (top_crop, bottom_crop, strategy) for tio.Crop on the H axis (dim 2).

    Small/medium masks (bbox shorter than crop): v3-style top-biased padding.
    Large masks (bbox taller than crop): superior anchor — bbox top sits near
    the top of the window; inferior mask extent is trimmed first (v2-like).
    Between the two, linearly blend crop start between center and superior.
    """
    if image_height <= target_height:
        return 0, 0, "no_crop"

    bbox_top = int(bbox_top)
    bbox_bottom = int(bbox_bottom)
    bbox_h = bbox_bottom - bbox_top + 1

    # Negative anchor → positive shift down (inferior); see LARGE_BBOX_SUPERIOR_ANCHOR.
    top_inset = int(round(target_height * large_bbox_superior_anchor))
    superior_start = bbox_top - top_inset

    blend_lo = int(target_height * large_bbox_blend_start_frac)
    extra = target_height - bbox_h
    top_pad = min(
        int(extra * top_pad_fraction),
        extra,
        max(12, int(0.08 * bbox_h)),
    )
    small_start = bbox_top - top_pad

    if bbox_h >= target_height:
        crop_start = superior_start
        strategy = "large_superior_anchor"
    elif bbox_h <= blend_lo:
        crop_start = small_start
        strategy = "small_top_pad"
    else:
        t = (bbox_h - blend_lo) / max(1, target_height - blend_lo)
        crop_start = int(round((1.0 - t) * small_start + t * superior_start))
        strategy = "blend"

    crop_start = max(0, min(crop_start, image_height - target_height))
    crop_end = crop_start + target_height

    if bbox_h >= target_height:
        # Large mask: keep superior edge; do not shift down to include inferior mask.
        if bbox_top < crop_start:
            crop_start = max(0, bbox_top - top_inset)
            crop_start = min(crop_start, image_height - target_height)
    else:
        # Small/medium: fit full bbox when possible (v3).
        if bbox_top < crop_start:
            crop_start = bbox_top
        if bbox_bottom >= crop_end:
            crop_start = bbox_bottom - target_height + 1
        crop_start = max(0, min(crop_start, image_height - target_height))

    top_crop = crop_start
    bottom_crop = image_height - crop_start - target_height
    return top_crop, bottom_crop, strategy


def get_breast_crop_bounds_from_mask(
    mask_3d: np.ndarray,
    image_height: int,
    target_height: int = 256,
    image_3d: np.ndarray | None = None,
    *,
    min_cc_fraction: float = MASK_CC_MIN_FRACTION,
    superior_bright_rows: int = MASK_CC_SUPERIOR_BRIGHT_ROWS,
) -> dict:
    """Vertical crop parameters from cleaned 3D mask (new_penn step2b).

    Returns a dict with ``vertical_top_crop``, ``vertical_bottom_crop``,
    ``mask_bbox_h_top`` / ``mask_bbox_h_bottom`` (split-side H indices),
    ``crop_method`` (``mask_bbox`` | ``center_fallback``), and
    ``vertical_h_start`` / ``vertical_h_end`` (half-open kept range on H).
    """
    mask_3d = np.asarray(mask_3d)
    if mask_3d.ndim != 3:
        raise ValueError(f"mask_3d must be 3D (W, H, D), got shape {mask_3d.shape}")

    cleaned = _keep_largest_3d_component(mask_3d, min_fraction=min_cc_fraction)
    extent = _vertical_h_extent(
        cleaned,
        image_3d=image_3d,
        superior_bright_rows=superior_bright_rows,
    )
    if extent is None:
        crop_start = (image_height - target_height) // 2
        top_crop = max(0, crop_start)
        bottom_crop = max(0, image_height - target_height - top_crop)
        crop_method = "center_fallback"
        bbox_top = bbox_bottom = None
    else:
        top_crop, bottom_crop, v_strategy = _vertical_crop_bounds(
            extent[0], extent[1], image_height, target_height
        )
        crop_method = f"mask_bbox_{v_strategy}"
        bbox_top, bbox_bottom = int(extent[0]), int(extent[1])

    h_end = image_height - bottom_crop
    bbox_h = (bbox_bottom - bbox_top + 1) if bbox_top is not None else None
    return {
        "crop_method": crop_method,
        "mask_bbox_h_top": bbox_top,
        "mask_bbox_h_bottom": bbox_bottom,
        "mask_bbox_h": bbox_h,
        "vertical_top_crop": int(top_crop),
        "vertical_bottom_crop": int(bottom_crop),
        "vertical_h_start": int(top_crop),
        "vertical_h_end": int(h_end),
    }


def crop_transform_from_bounds(bounds: dict) -> tio.Crop:
    """Build ``tio.Crop`` from ``get_breast_crop_bounds_from_mask`` output."""
    return tio.Crop(
        (0, 0, bounds["vertical_top_crop"], bounds["vertical_bottom_crop"], 0, 0)
    )


def get_breast_crop_transform_from_mask(
    mask_3d: np.ndarray,
    image_height: int,
    target_height: int = 256,
    image_3d: np.ndarray | None = None,
    *,
    min_cc_fraction: float = MASK_CC_MIN_FRACTION,
    superior_bright_rows: int = MASK_CC_SUPERIOR_BRIGHT_ROWS,
):
    """Vertical crop from cleaned 3D mask bbox (new_penn step2b).

    1. Keep largest 3D mask component (drops small island artifacts).
    2. Vertical extent from cleaned mask; optionally extend superior edge
       with bright voxels in ``image_3d`` (pre) within a band above the mask.
    3. Top-biased padding inside the crop window.
    """
    bounds = get_breast_crop_bounds_from_mask(
        mask_3d,
        image_height,
        target_height,
        image_3d,
        min_cc_fraction=min_cc_fraction,
        superior_bright_rows=superior_bright_rows,
    )
    return crop_transform_from_bounds(bounds)


def _crop_or_pad_axis_range(size: int, target: int) -> dict:
    """How one spatial axis maps through ``tio.CropOrPad`` (center crop / symmetric pad)."""
    size, target = int(size), int(target)
    if size >= target:
        start = (size - target) // 2
        return {
            "input_start": start,
            "input_end": start + target,
            "pad_before": 0,
            "pad_after": 0,
        }
    pad_before = (target - size) // 2
    pad_after = target - size - pad_before
    return {
        "input_start": 0,
        "input_end": size,
        "pad_before": pad_before,
        "pad_after": pad_after,
    }


def build_crop_coordinate_record(
    *,
    patient_id: str,
    side: str,
    ref_shape: tuple[int, int, int],
    vertical_bounds: dict,
    target_shape: tuple[int, int, int],
    processing_mode: str = "slice",
    step1_rot90_k: int = 0,
) -> dict:
    """Map step2b crops to voxel indices in the reference ``pre.nii.gz``.

    TorchIO layout is ``(C, W, H, D)``. Indices ``full_*`` are half-open
    ranges in that preprocessed volume (after step1, before step2b). To map
    an output voxel ``(w, h, d)`` back when ``pad_*`` is zero::

        w_full = full_w_start + w
        h_full = full_h_start + h
        d_full = full_d_start + d

    When ``pad_*`` > 0, output voxels in the padded margin have no source voxel.
    """
    W, H, D = (int(ref_shape[0]), int(ref_shape[1]), int(ref_shape[2]))
    # step2b CLI / DEFAULT_TARGET_SHAPE use (H, W, D); TorchIO tensors are (W, H, D).
    target_h, target_w, target_d = (
        int(target_shape[0]),
        int(target_shape[1]),
        int(target_shape[2]),
    )

    if side == "left":
        split_crop_left, split_crop_right = 0, W // 2
        split_w_start = 0
        split_w = W - W // 2
    elif side == "right":
        split_crop_left, split_crop_right = W // 2, 0
        split_w_start = W // 2
        split_w = W - W // 2
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    top = int(vertical_bounds["vertical_top_crop"])
    bottom = int(vertical_bounds["vertical_bottom_crop"])
    h_after_v = H - top - bottom

    w_rng = _crop_or_pad_axis_range(split_w, target_w)
    h_rng = _crop_or_pad_axis_range(h_after_v, target_h)
    d_rng = _crop_or_pad_axis_range(D, target_d)

    full_w_start = split_w_start + w_rng["input_start"]
    full_w_end = split_w_start + w_rng["input_end"]
    full_h_start = top + h_rng["input_start"]
    full_h_end = top + h_rng["input_end"]
    full_d_start = d_rng["input_start"]
    full_d_end = d_rng["input_end"]

    return {
        "patient_id": patient_id,
        "side": side,
        "output_uid": f"{patient_id}_{side}",
        "processing_mode": processing_mode,
        "coordinate_space": "step1_pre_nifti_WH_D",
        "step1_rot90_k": int(step1_rot90_k),
        "ref_shape_W": W,
        "ref_shape_H": H,
        "ref_shape_D": D,
        "split_crop_left": split_crop_left,
        "split_crop_right": split_crop_right,
        "split_shape_W": split_w,
        "split_shape_H": H,
        "split_shape_D": D,
        "crop_method": vertical_bounds["crop_method"],
        "mask_bbox_h_top": vertical_bounds["mask_bbox_h_top"],
        "mask_bbox_h_bottom": vertical_bounds["mask_bbox_h_bottom"],
        "vertical_top_crop": top,
        "vertical_bottom_crop": bottom,
        "vertical_h_start": int(vertical_bounds["vertical_h_start"]),
        "vertical_h_end": int(vertical_bounds["vertical_h_end"]),
        "after_vertical_shape_W": split_w,
        "after_vertical_shape_H": h_after_v,
        "after_vertical_shape_D": D,
        "crop_or_pad_w_input_start": w_rng["input_start"],
        "crop_or_pad_w_input_end": w_rng["input_end"],
        "crop_or_pad_w_pad_before": w_rng["pad_before"],
        "crop_or_pad_w_pad_after": w_rng["pad_after"],
        "crop_or_pad_h_input_start": h_rng["input_start"],
        "crop_or_pad_h_input_end": h_rng["input_end"],
        "crop_or_pad_h_pad_before": h_rng["pad_before"],
        "crop_or_pad_h_pad_after": h_rng["pad_after"],
        "crop_or_pad_d_input_start": d_rng["input_start"],
        "crop_or_pad_d_input_end": d_rng["input_end"],
        "crop_or_pad_d_pad_before": d_rng["pad_before"],
        "crop_or_pad_d_pad_after": d_rng["pad_after"],
        "target_shape_H": target_h,
        "target_shape_W": target_w,
        "target_shape_D": target_d,
        "full_w_start": full_w_start,
        "full_w_end": full_w_end,
        "full_h_start": full_h_start,
        "full_h_end": full_h_end,
        "full_d_start": full_d_start,
        "full_d_end": full_d_end,
    }


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
            top_crop, bottom_crop, _ = _vertical_crop_bounds(
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