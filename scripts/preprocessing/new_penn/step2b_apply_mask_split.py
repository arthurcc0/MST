"""Step 2b: mask + canonicalize + split into L/R + crop/pad to (256, 256, 32).

Mirrors ``scripts/preprocessing/penn-preprocessed/run_new_pipeline.py`` (image
x mask, split L/R, breast-localized crop, pad to fixed shape) but adapted for
the new_penn layout:

    - masks live under ``MASK_DIR`` as ``<newaccession>.npz`` with key ``'data'``
      and have the same shape as the volumes;
    - we reuse ``get_breast_crop_transform`` from
      ``penn-preprocessed/step2b_crop_or_pad.py`` instead of duplicating it.

For each case this writes::

    <OUT_ROOT>/<PatientID>_left/<image_name>
    <OUT_ROOT>/<PatientID>_right/<image_name>

for every ``*.nii.gz`` file in the case's folder (typically pre, post, sub).
"""

import functools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from scipy.ndimage import rotate
from tqdm import tqdm

# Reuse the robust breast-crop helper from the existing penn-preprocessed pipeline.
sys.path.append(str(Path(__file__).resolve().parents[1] / "penn-preprocessed"))
from step2b_crop_or_pad import get_breast_crop_transform  # noqa: E402

# ---- Configurable paths and parameters --------------------------------------
OUT_ROOT_BASE = Path(r"D:\Users\arthur\Data\MST_birads4")
IN_DATA_ROOT = OUT_ROOT_BASE / "preprocessed" / "data"  # output of step2a
MASK_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_masks\breast")
OUT_ROOT = OUT_ROOT_BASE / "final_cropped_and_masked_data"

MASK_KEY = "data"  # key inside the mask .npz
TARGET_SHAPE = (256, 256, 32)
TARGET_HEIGHT = 256
ROTATE_MASK_DEG = 0  # set to 90 if mask orientation differs from the volume

# Mirror step1's ROT90_K constant directly to avoid pulling step1_npz2nifti
# (and therefore torch) into worker spawn-time. If you change ROT90_K in
# step1, mirror it here.
STEP1_ROT90_K = 1

# Number of parallel workers. Each worker imports torch/torchio/scipy at
# spawn time on Windows, which is heavy; do not use Pool() (= one per core).
NUM_WORKERS = 2

# Debug: process only the first case
DEBUG_SINGLE = False

def _load_mask_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        if MASK_KEY in npz.files:
            arr = np.asarray(npz[MASK_KEY])
            # Binarize at load time (matches the convention used in step2a:
            # ``(mask > 0.5).astype(uint8)``). TorchIO uses nearest-neighbor
            # for LabelMap transforms, so this stays binary downstream.
            return (arr > 0.5).astype(np.uint8)
        raise KeyError(f"Key '{MASK_KEY}' not found in {path}; got {list(npz.files)}")


def _process_case(path_dir: Path, in_root_str: str, mask_dir_str: str, out_root_str: str) -> None:
    in_root = Path(in_root_str)  # noqa: F841 (reserved for future relative-path use)
    mask_dir = Path(mask_dir_str)
    out_root = Path(out_root_str)
    patient_id = path_dir.name

    # Skip if every input nifti already has a corresponding output on both sides.
    input_names = [p.name for p in path_dir.glob("*.nii.gz")]
    if input_names:
        out_left = out_root / f"{patient_id}_left"
        out_right = out_root / f"{patient_id}_right"
        if out_left.is_dir() and out_right.is_dir():
            existing_left = {p.name for p in out_left.glob("*.nii.gz")}
            existing_right = {p.name for p in out_right.glob("*.nii.gz")}
            needed = set(input_names)
            if needed.issubset(existing_left) and needed.issubset(existing_right):
                return

    mask_path = mask_dir / f"{patient_id}.npz"
    if not mask_path.is_file():
        return

    try:
        mask_arr = _load_mask_npz(mask_path)
        if mask_arr.size == 0:
            return

        # Use pre.nii.gz as the geometric reference for the mask.
        pre_path = path_dir / "pre.nii.gz"
        if not pre_path.exists():
            return
        pre_img = tio.ScalarImage(pre_path)

        # Step1 wrote NIfTIs with identity-direction affines and applied a
        # torch.rot90(k=ROT90_K, dims=(0,1)) to the volume. Masks come from
        # the original .npz layout, so we apply the same rotation to keep
        # them aligned with the image. We do NOT call tio.ToCanonical here:
        # our affines are already RAS-with-spacing, and a redundant
        # canonicalization on only the mask was causing it to end up rotated
        # 90 degrees relative to the image (the cause of empty-side cases
        # like 59206900_left).
        if STEP1_ROT90_K:
            mask_arr = np.rot90(mask_arr, k=STEP1_ROT90_K, axes=(0, 1)).copy()
        if ROTATE_MASK_DEG:
            mask_arr = rotate(mask_arr, angle=ROTATE_MASK_DEG, axes=(0, 1), reshape=False)
        mask_data = mask_arr.astype(np.uint8)
        del mask_arr

        # ---- Pre-compute the per-side breast crop on the reference image ----
        width = pre_img.shape[1]
        split_transforms = {
            "right": tio.Crop((width // 2, 0, 0, 0, 0, 0)),
            "left": tio.Crop((0, width // 2, 0, 0, 0, 0)),
        }
        crop_transforms = {
            side: get_breast_crop_transform(split_transforms[side](pre_img), target_height=TARGET_HEIGHT)
            for side in ("left", "right")
        }
        del pre_img
        pad_transform = tio.CropOrPad(TARGET_SHAPE, padding_mode=0)

        # ---- Process every image in this case folder (pre, post, sub) ----
        for path_img in path_dir.glob("*.nii.gz"):
            img = tio.ScalarImage(path_img)
            if img.shape[1] < 100:
                del img
                continue

            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img.data, affine=img.affine),
                mask=tio.LabelMap(
                    tensor=torch.from_numpy(mask_data).unsqueeze(0), affine=img.affine
                ),
            )
            del img

            for side in ("left", "right"):
                side_subject = split_transforms[side](subject)
                final_transform = tio.Compose([crop_transforms[side], pad_transform])
                processed = final_transform(side_subject)
                del side_subject

                masked = (
                    processed.image.data.squeeze().numpy()
                    * processed.mask.data.squeeze().numpy()
                )
                out_dir = out_root / f"{patient_id}_{side}"
                out_dir.mkdir(parents=True, exist_ok=True)
                tio.ScalarImage(
                    tensor=torch.from_numpy(masked).unsqueeze(0),
                    affine=processed.image.affine,
                ).save(out_dir / path_img.name)
                del processed, masked
            del subject

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")


def main() -> None:
    if not IN_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Input data root not found: {IN_DATA_ROOT}")
    if not MASK_DIR.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    patient_dirs = [p for p in IN_DATA_ROOT.iterdir() if p.is_dir()]
    if DEBUG_SINGLE:
        patient_dirs = patient_dirs[:1]
        print(f"[DEBUG_SINGLE] only processing: {[p.name for p in patient_dirs]}")
    print(f"Processing {len(patient_dirs)} cases. Output: {OUT_ROOT}")

    processor = functools.partial(
        _process_case,
        in_root_str=str(IN_DATA_ROOT),
        mask_dir_str=str(MASK_DIR),
        out_root_str=str(OUT_ROOT),
    )

    if DEBUG_SINGLE:
        for p in patient_dirs:
            processor(p)
    else:
        with Pool(processes=NUM_WORKERS) as pool:
            for _ in tqdm(pool.imap_unordered(processor, patient_dirs), total=len(patient_dirs)):
                pass

    print("Mask + crop/split done.")


if __name__ == "__main__":
    main()
