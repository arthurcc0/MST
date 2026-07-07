"""Step 1: convert per-case .npz volumes to per-case NIfTI files.

For each row in ``new_penn_mapping.csv`` this script loads the .npz file once
and writes::

    <OUT_ROOT>/preprocessed/data/<PatientID>/pre.nii.gz
    <OUT_ROOT>/preprocessed/data/<PatientID>/post.nii.gz

The .npz contains:
    - ``pre``     : pre-contrast volume (3D array)
    - ``post``    : post-contrast volume (3D array, same shape as ``pre``)
    - ``dic_pre`` : DICOM-like metadata dict for ``pre`` (pickled, holds spacing)
    - ``dic_post``: DICOM-like metadata dict for ``post``

Voxel spacing is pulled from ``dic_pre`` (PixelSpacing + SliceThickness /
SpacingBetweenSlices). If unavailable we fall back to ``DEFAULT_SPACING``.

The ``ROT90_K`` toggle mirrors what penn-preprocessed/step1npy2nift.py did so
``step2b`` downstream (which uses ``width // 2`` to split L/R) sees the
expected orientation. Set ``ROT90_K = 0`` if your arrays are already correct.
"""

import functools
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchio as tio
from tqdm import tqdm

# ---- Configurable paths and parameters --------------------------------------
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
OUT_DATA_DIR = OUT_ROOT / "preprocessed" / "data"

# For new Penn data (BIRADS-4):
MAPPING_CSV = OUT_ROOT / "new_penn_mapping_v2.csv"

# For old Penn data (all BI-RADS):
# MAPPING_CSV = OUT_ROOT / "old_penn_mapping_v2.csv"

PRE_KEY = "pre"
POST_KEY = "post"
DIC_PRE_KEY = "dic_pre"
DIC_POST_KEY = "dic_post"

DEFAULT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
ROT90_K = 1  # 90-deg rotations on dims (0, 1); 0 disables

# Number of parallel workers. Each worker transiently holds ~3-4 copies of
# one volume in memory, so don't blindly use Pool() (= one per core); on a
# 16-core machine that easily OOMs for ~80 MB float32 volumes. Tune up if
# you have RAM to spare.
NUM_WORKERS = 2

# Debug: process only the first case
DEBUG_SINGLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def _save_with_timeout(img: tio.ScalarImage, path: Path, timeout: int = 60) -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(img.save, path)
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            raise Exception(f"Save operation timed out after {timeout} seconds for {path}")


def _spacing_from_dic(dic: dict) -> Tuple[float, float, float]:
    """Extract (sx, sy, sz) in mm from a DICOM-like metadata dict.

    Falls back to ``DEFAULT_SPACING`` for any missing field. PixelSpacing is
    [row_mm, col_mm] in DICOM convention; we map (row, col) -> (sx, sy).
    """
    sx, sy, sz = DEFAULT_SPACING
    try:
        ps = dic.get("PixelSpacing")
        if ps is not None and len(ps) >= 2:
            sx = float(ps[0])
            sy = float(ps[1])
    except Exception:
        pass

    for key in ("SpacingBetweenSlices", "SliceThickness"):
        try:
            v = dic.get(key)
            if v is not None:
                sz = float(v)
                break
        except Exception:
            continue

    return (sx, sy, sz)


def _array_to_scalar_image(arr: np.ndarray, spacing: Tuple[float, float, float]) -> tio.ScalarImage:
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}")
    tensor = torch.from_numpy(arr)  # zero-copy if contiguous; rot90 below makes a contiguous copy anyway
    if ROT90_K:
        tensor = torch.rot90(tensor, k=ROT90_K, dims=(0, 1)).contiguous()
    tensor = tensor.unsqueeze(0)  # (C, X, Y, Z)
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    return tio.ScalarImage(tensor=tensor, affine=affine)


def _process_row(row: dict, out_data_dir_str: str) -> None:
    out_data_dir = Path(out_data_dir_str)
    patient_id = str(row["PatientID"])
    npz_path = Path(row["FullPath"])

    if not npz_path.is_file():
        logger.warning(f"Missing file for {patient_id}: {npz_path}")
        return

    # Skip if a previous run already produced both outputs for this case.
    out_dir = out_data_dir / patient_id
    if (out_dir / "pre.nii.gz").exists() and (out_dir / "post.nii.gz").exists():
        return

    try:
        # Stream pre and post sequentially so a worker only holds the copies
        # of ONE volume at a time (each transient peak ~3 x volume size).
        with np.load(npz_path, allow_pickle=True) as npz:
            keys = list(npz.files)
            if PRE_KEY not in keys or POST_KEY not in keys:
                logger.warning(f"{patient_id}: expected keys '{PRE_KEY}'/'{POST_KEY}', got {keys}")
                return

            dic_pre = {}
            if DIC_PRE_KEY in keys:
                try:
                    dic_pre = npz[DIC_PRE_KEY].item()
                except Exception:
                    dic_pre = {}
            spacing = _spacing_from_dic(dic_pre) if isinstance(dic_pre, dict) else DEFAULT_SPACING

            out_dir.mkdir(parents=True, exist_ok=True)

            pre_arr = np.asarray(npz[PRE_KEY])
            pre_img = _array_to_scalar_image(pre_arr, spacing)
            del pre_arr
            _save_with_timeout(pre_img, out_dir / "pre.nii.gz")
            del pre_img

            post_arr = np.asarray(npz[POST_KEY])
            post_img = _array_to_scalar_image(post_arr, spacing)
            del post_arr
            _save_with_timeout(post_img, out_dir / "post.nii.gz")
            del post_img

    except Exception:
        logger.error(f"Failed to process {patient_id} ({npz_path}):\n{traceback.format_exc()}")


def main() -> None:
    if not MAPPING_CSV.is_file():
        raise FileNotFoundError(
            f"Mapping CSV not found at {MAPPING_CSV}. Run new_penn_mapping.py first."
        )

    df_mapping = pd.read_csv(MAPPING_CSV, dtype={"PatientID": str})
    df_mapping = df_mapping.dropna(subset=["FullPath"]).drop_duplicates(subset=["PatientID"])

    if DEBUG_SINGLE:
        df_mapping = df_mapping.head(1)
        logger.info(f"[DEBUG_SINGLE] only converting: {df_mapping['PatientID'].tolist()}")

    rows = df_mapping[["PatientID", "FullPath"]].to_dict(orient="records")
    logger.info(f"Converting {len(rows)} cases to NIfTI under {OUT_DATA_DIR}")

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processor = functools.partial(_process_row, out_data_dir_str=str(OUT_DATA_DIR))

    if DEBUG_SINGLE:
        for row in rows:
            processor(row)
    else:
        with Pool(processes=NUM_WORKERS) as pool:
            for _ in tqdm(pool.imap_unordered(processor, rows), total=len(rows)):
                pass

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
