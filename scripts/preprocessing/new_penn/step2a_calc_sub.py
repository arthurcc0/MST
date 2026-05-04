"""Step 2a: write per-case ``sub.nii.gz`` = post - (pre * breast_mask).

The breast mask (``<MASK_DIR>/<PatientID>.npz`` with key ``'data'``) is
applied to ``pre`` *before* the subtraction so that voxels outside the breast
do not contribute background signal to the subtracted volume. ``post`` is
left unmasked at this stage; full image-level masking is done later in
``step2b_apply_mask_split.py``.

Notes
-----
The mask in the .npz lives in the raw array space, not the rotated /
canonical space written by step1. We therefore apply the same ``ROT90_K``
rotation as step1 to bring it into alignment with ``pre.nii.gz`` before
multiplying.
"""

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ---- Configurable paths -----------------------------------------------------
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
DATA_ROOT = OUT_ROOT / "preprocessed" / "data"  # output of step1
MASK_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_masks\breast")
MASK_KEY = "data"

# Pull the rotation step1 applied so the mask aligns with pre/post. We
# intentionally do NOT import step1_npz2nifti here, because that would also
# import torch into every worker (which has caused c10.dll crashes on Windows
# spawn workers). If you change ROT90_K in step1, mirror it here.
STEP1_ROT90_K = 1

# Number of parallel workers. Each worker transiently holds ~5 copies of the
# volume (~80 MB each) so keep this conservative; raise if RAM allows.
NUM_WORKERS = 2

# Debug: process only the first case
DEBUG_SINGLE = False

logger = logging.getLogger(__name__)


def _load_mask(patient_id: str) -> np.ndarray | None:
    mask_path = MASK_DIR / f"{patient_id}.npz"
    if not mask_path.is_file():
        return None
    with np.load(mask_path, allow_pickle=True) as npz:
        if MASK_KEY not in npz.files:
            logger.warning(f"{patient_id}: key '{MASK_KEY}' not in mask .npz (got {list(npz.files)})")
            return None
        arr = np.asarray(npz[MASK_KEY])
    if STEP1_ROT90_K:
        # Match the rotation that step1 applied to pre/post in numpy axes (0, 1).
        # Pure-numpy version (no torch) avoids loading the c10 runtime in workers.
        arr = np.rot90(arr, k=STEP1_ROT90_K, axes=(0, 1)).copy()
    return arr


def process(path_patient: Path) -> None:
    patient_id = path_patient.name
    path_pre = path_patient / "pre.nii.gz"
    path_post = path_patient / "post.nii.gz"
    path_sub = path_patient / "sub.nii.gz"

    if path_sub.exists():
        return

    if not path_pre.exists() or not path_post.exists():
        logger.warning(f"Skipping {patient_id}: missing pre or post.")
        return

    try:
        # ---- Load mask first; bail out early if missing/mismatched ---------
        mask_arr = _load_mask(patient_id)
        if mask_arr is None:
            logger.warning(f"{patient_id}: no mask found; skipping (sub not written).")
            return

        pre_nii = sitk.ReadImage(str(path_pre), sitk.sitkInt32)
        post_nii_orig = sitk.ReadImage(str(path_post), sitk.sitkInt32)

        post_nii = sitk.Resample(
            post_nii_orig, pre_nii, sitk.Transform(), sitk.sitkLinear, 0, pre_nii.GetPixelID()
        )
        del post_nii_orig

        pre_arr = sitk.GetArrayFromImage(pre_nii).astype(np.int32)
        post_arr = sitk.GetArrayFromImage(post_nii).astype(np.int32)
        del post_nii  # we already have the resampled array

        # SimpleITK arrays are (Z, Y, X); the mask is in the original (X, Y, Z)
        # layout, so transpose if needed.
        if mask_arr.shape != pre_arr.shape:
            if mask_arr.shape[::-1] == pre_arr.shape:
                mask_arr = np.transpose(mask_arr, (2, 1, 0))
            else:
                logger.warning(
                    f"{patient_id}: mask shape {mask_arr.shape} doesn't match pre {pre_arr.shape}; skipping."
                )
                return

        mask_bin = (mask_arr > 0.5).astype(np.uint8)
        del mask_arr

        # In-place ops where safe to keep the peak low.
        pre_arr *= mask_bin  # pre <- pre * mask
        del mask_bin
        sub_arr = post_arr
        sub_arr -= pre_arr  # sub <- post - (pre * mask), in place on post_arr buffer
        del pre_arr, post_arr

        sub_arr -= sub_arr.min()
        sub_arr = sub_arr.astype(np.uint16)

        sub_nii = sitk.GetImageFromArray(sub_arr)
        sub_nii.CopyInformation(pre_nii)
        del sub_arr, pre_nii

        sitk.WriteImage(sub_nii, str(path_sub))
        del sub_nii
    except Exception as e:
        logger.error(f"Failed to process {patient_id}: {e}")


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
    if not MASK_DIR.is_dir():
        raise FileNotFoundError(f"Mask dir not found: {MASK_DIR}")

    patient_dirs = [p for p in DATA_ROOT.iterdir() if p.is_dir()]
    if DEBUG_SINGLE:
        patient_dirs = patient_dirs[:1]
        print(f"[DEBUG_SINGLE] only processing: {[p.name for p in patient_dirs]}")

    print(f"Processing subtraction for {len(patient_dirs)} cases.")

    if DEBUG_SINGLE:
        for p in patient_dirs:
            process(p)
    else:
        with Pool(processes=NUM_WORKERS) as pool:
            for _ in tqdm(pool.imap_unordered(process, patient_dirs), total=len(patient_dirs)):
                pass

    print("Subtraction calculation complete.")


if __name__ == "__main__":
    main()
