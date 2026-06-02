"""Step 2b: mask + canonicalize + split into L/R + crop/pad to (H, W, D).

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

Two depth-handling modes are supported:

* **slice mode (default)** — final shape is ``(H, W, D)`` taken from
  ``--height/--width/--depth``. Output folder is
  ``final_cropped_and_masked_data`` (or ``..._d{D}`` when D != 32).
* **slab mode (``--slabs``)** — each output slice is a maximum-intensity
  projection (MIP) of ``--slab_size`` adjacent slices, advanced by
  ``--slab_size - --overlap`` slices between slabs, and centered along the
  depth axis. Mask is applied **before** MIP so background voxels can't
  dominate ``max``. When the source depth is insufficient to fit
  ``--num_slabs`` slabs, the available slabs are still centered and the
  first/last slab are repeated to reach ``--num_slabs``. Output folder is
  ``final_cropped_and_masked_slabs_n{N}_s{S}_o{O}``.
"""

import argparse
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
from step2b_crop_or_pad import get_breast_crop_transform_from_mask  # noqa: E402

# ---- Configurable paths and parameters --------------------------------------
OUT_ROOT_BASE = Path(r"D:\Users\arthur\Data\MST_birads4")
IN_DATA_ROOT = OUT_ROOT_BASE / "preprocessed" / "data"  # output of step2a
MASK_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_masks\breast")

# Defaults — overridable via CLI. The 32-slice default keeps the existing
# OUT_ROOT name unsuffixed for backward compatibility.
DEFAULT_TARGET_SHAPE = (256, 256, 32)
TARGET_SHAPE = DEFAULT_TARGET_SHAPE
TARGET_HEIGHT = TARGET_SHAPE[0]
OUT_ROOT = OUT_ROOT_BASE / "final_cropped_and_masked_data"

# Slab-mode defaults. Activated when SLAB_PARAMS is not None (driven by --slabs).
DEFAULT_SLAB_PARAMS = dict(num_slabs=32, slab_size=3, overlap=0)

MASK_KEY = "data"  # key inside the mask .npz
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


def _compute_slabs(volume_3d: np.ndarray, num_slabs: int, slab_size: int, overlap: int):
    """Compute MIP slabs along the last axis.

    Each output slab is ``volume_3d[..., a:a+slab_size].max(axis=-1)``. The
    window of slabs is centered on the depth axis. If the source depth ``D``
    cannot fit ``num_slabs`` slabs at the requested overlap, we compute as
    many as fit (still centered) and then repeat the first/last slab to
    pad the output up to ``num_slabs``.

    Returns
    -------
    out : np.ndarray of shape ``volume_3d.shape[:-1] + (num_slabs,)``,
          or ``None`` when ``D < slab_size`` (cannot build a single slab).
    record : dict with keys ``status`` ('ok' | 'padded_edges' | 'too_shallow'),
             ``depth`` (source D), ``k_native`` (slabs computed before edge
             repeats), ``pre_pad`` and ``post_pad`` (edge-repeat counts).
    """
    S, N, O = int(slab_size), int(num_slabs), int(overlap)
    stride = S - O
    if stride < 1:
        raise ValueError(f"overlap must be < slab_size; got slab_size={S}, overlap={O}")

    D = int(volume_3d.shape[-1])
    record = {"depth": D, "k_native": 0, "pre_pad": 0, "post_pad": 0, "status": "ok"}

    if D < S:
        record["status"] = "too_shallow"
        return None, record

    # max_fit is how many slabs of size S, stride `stride`, fit in D slices.
    max_fit = (D - S) // stride + 1
    k = min(N, max_fit)

    # window_len is the span the k slabs occupy together, i.e. from the first
    # slice of slab 0 to the last slice of slab k-1 inclusive. With overlap >= 0
    # the slabs are contiguous, so window_len equals the number of unique
    # source slices touched.
    window_len = (k - 1) * stride + S

    # Center the WHOLE window in [0, D) -- not just the first slab. Leftover
    # slices are split as evenly as possible: floor((D-window_len)/2) on the
    # left and ceil((D-window_len)/2) on the right.
    start = (D - window_len) // 2

    slabs = [volume_3d[..., start + i * stride : start + i * stride + S].max(axis=-1) for i in range(k)]

    if k < N:
        # Fewer slabs fit than requested. Keep the native slabs centered in the
        # source volume and pad to N by repeating the first/last slab; the
        # repeats themselves are also split as evenly as possible around the
        # native block, so the final stack of N slabs stays centered.
        n_missing = N - k
        pre = n_missing // 2
        post = n_missing - pre
        slabs = [slabs[0]] * pre + slabs + [slabs[-1]] * post
        record.update(status="padded_edges", pre_pad=pre, post_pad=post)

    record["k_native"] = k
    return np.stack(slabs, axis=-1), record


def _process_case(
    path_dir: Path,
    in_root_str: str,
    mask_dir_str: str,
    out_root_str: str,
    target_shape: tuple = TARGET_SHAPE,
    target_height: int = TARGET_HEIGHT,
    slab_params: dict = None,
) -> list:
    """Process a single case. Returns a list of per-(image, side) stats records.

    When ``slab_params`` is None (default), behaves like the original
    slice-mode pipeline. When provided (``num_slabs``, ``slab_size``,
    ``overlap``), each output slice is a MIP slab of ``slab_size`` adjacent
    source slices.
    """
    in_root = Path(in_root_str)  # noqa: F841 (reserved for future relative-path use)
    mask_dir = Path(mask_dir_str)
    out_root = Path(out_root_str)
    patient_id = path_dir.name
    stats_out: list = []

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
                return stats_out

    mask_path = mask_dir / f"{patient_id}.npz"
    if not mask_path.is_file():
        return stats_out

    try:
        mask_arr = _load_mask_npz(mask_path)
        if mask_arr.size == 0:
            return stats_out

        # Use pre.nii.gz as the geometric reference for the mask.
        pre_path = path_dir / "pre.nii.gz"
        if not pre_path.exists():
            return stats_out
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

        # ---- Pre-compute per-side vertical crop from the breast mask bbox ----
        # Using intensity-based centering (get_breast_crop_transform) tended to
        # shift the window inferiorly, clipping the superior breast and leaving
        # background at the bottom. The mask bbox + top-biased padding keeps
        # the full ROI in frame.
        width = pre_img.shape[1]
        split_transforms = {
            "right": tio.Crop((width // 2, 0, 0, 0, 0, 0)),
            "left": tio.Crop((0, width // 2, 0, 0, 0, 0)),
        }
        mask_label = tio.LabelMap(
            tensor=torch.from_numpy(mask_data).unsqueeze(0), affine=pre_img.affine
        )
        crop_transforms = {}
        for side in ("left", "right"):
            side_mask = split_transforms[side](mask_label)
            side_h = split_transforms[side](pre_img).shape[2]
            crop_transforms[side] = get_breast_crop_transform_from_mask(
                side_mask.data.squeeze().numpy(),
                image_height=side_h,
                target_height=target_height,
            )
        del pre_img, mask_label

        if slab_params is None:
            pad_transform = tio.CropOrPad(target_shape, padding_mode=0)
            inplane_pad = None
        else:
            pad_transform = None
            inplane_target = (int(target_shape[0]), int(target_shape[1]), int(slab_params["num_slabs"]))
            inplane_pad = tio.CropOrPad(inplane_target, padding_mode=0)

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

                if slab_params is None:
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
                else:
                    # Breast-crop H only; depth stays at the source D.
                    breast_cropped = crop_transforms[side](side_subject)
                    del side_subject

                    img_arr = breast_cropped.image.data.squeeze().numpy().astype(np.float32)
                    msk_arr = breast_cropped.mask.data.squeeze().numpy().astype(np.float32)
                    masked_vol = img_arr * msk_arr  # mask before MIP
                    del img_arr, msk_arr

                    slabs, record = _compute_slabs(masked_vol, **slab_params)
                    record.update(patient_id=patient_id, side=side, image=path_img.name)
                    stats_out.append(record)
                    del masked_vol

                    if slabs is None:
                        # too shallow for even one slab: skip writing this side+image.
                        del breast_cropped
                        continue

                    slab_tensor = torch.from_numpy(np.ascontiguousarray(slabs)).unsqueeze(0)
                    slab_subject = tio.Subject(
                        image=tio.ScalarImage(tensor=slab_tensor, affine=breast_cropped.image.affine)
                    )
                    del breast_cropped, slabs

                    processed = inplane_pad(slab_subject)
                    out_dir = out_root / f"{patient_id}_{side}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    tio.ScalarImage(
                        tensor=processed.image.data,
                        affine=processed.image.affine,
                    ).save(out_dir / path_img.name)
                    del slab_subject, processed
            del subject

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")

    return stats_out


def _summarize_slab_stats(records: list, out_csv: Path, slab_params: dict) -> None:
    """Write a per-(image, side) CSV and print a one-screen summary."""
    import csv
    from statistics import median, mean

    if not records:
        print("[slab] no slab records produced (nothing to summarize).")
        return

    fieldnames = ["patient_id", "side", "image", "depth", "status", "k_native", "pre_pad", "post_pad"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    ok = sum(1 for r in records if r["status"] == "ok")
    padded = sum(1 for r in records if r["status"] == "padded_edges")
    too_shallow = sum(1 for r in records if r["status"] == "too_shallow")
    depths = [r["depth"] for r in records]
    stride = slab_params["slab_size"] - slab_params["overlap"]
    primary_req = (slab_params["num_slabs"] - 1) * stride + slab_params["slab_size"]

    print(
        "\n[slab summary] (image, side) records:"
        f"\n  total:        {len(records)}"
        f"\n  ok:           {ok}    (D >= {primary_req})"
        f"\n  padded_edges: {padded} (D < {primary_req}; first/last slab repeated)"
        f"\n  too_shallow:  {too_shallow} (D < {slab_params['slab_size']}; skipped)"
        f"\n  source depth: min={min(depths)}, median={median(depths):.1f}, "
        f"mean={mean(depths):.1f}, max={max(depths)}"
        f"\n  details:      {out_csv}"
    )


def main(
    target_shape: tuple = TARGET_SHAPE,
    out_root: Path = OUT_ROOT,
    slab_params: dict = None,
) -> None:
    if not IN_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Input data root not found: {IN_DATA_ROOT}")
    if not MASK_DIR.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    patient_dirs = [p for p in IN_DATA_ROOT.iterdir() if p.is_dir()]
    if DEBUG_SINGLE:
        patient_dirs = patient_dirs[:1]
        print(f"[DEBUG_SINGLE] only processing: {[p.name for p in patient_dirs]}")
    mode_desc = (
        f"slab mode (N={slab_params['num_slabs']}, S={slab_params['slab_size']}, "
        f"O={slab_params['overlap']})"
        if slab_params is not None
        else f"slice mode (target_shape={target_shape})"
    )
    print(f"Processing {len(patient_dirs)} cases. Output: {out_root}  [{mode_desc}]")

    # Bind the depth-dependent shape into the worker via partial so it survives
    # process spawn on Windows (workers re-import the module without running __main__).
    processor = functools.partial(
        _process_case,
        in_root_str=str(IN_DATA_ROOT),
        mask_dir_str=str(MASK_DIR),
        out_root_str=str(out_root),
        target_shape=tuple(target_shape),
        target_height=int(target_shape[0]),
        slab_params=slab_params,
    )

    all_stats: list = []
    if DEBUG_SINGLE:
        for p in patient_dirs:
            all_stats.extend(processor(p) or [])
    else:
        with Pool(processes=NUM_WORKERS) as pool:
            for case_stats in tqdm(pool.imap_unordered(processor, patient_dirs), total=len(patient_dirs)):
                all_stats.extend(case_stats or [])

    if slab_params is not None:
        _summarize_slab_stats(all_stats, out_root / "slab_stats.csv", slab_params)

    print("Mask + crop/split done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mask + canonicalize + split L/R + crop/pad new_penn volumes.')
    parser.add_argument('--height', type=int, default=DEFAULT_TARGET_SHAPE[0], help='Target H (in-plane).')
    parser.add_argument('--width', type=int, default=DEFAULT_TARGET_SHAPE[1], help='Target W (in-plane).')
    parser.add_argument('--depth', type=int, default=DEFAULT_TARGET_SHAPE[2],
                        help='Target D (number of slices). Ignored in slab mode (D = --num_slabs).')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Override output directory. Default for slice mode: '
                             'OUT_ROOT_BASE/final_cropped_and_masked_data[_d{D}] when D != 32. '
                             'Default for slab mode: OUT_ROOT_BASE/final_cropped_and_masked_slabs_n{N}_s{S}_o{O}.')
    # --- Slab mode ---
    parser.add_argument('--slabs', action='store_true', default=False,
                        help='Activate slab (MIP) mode along the depth axis.')
    parser.add_argument('--num_slabs', type=int, default=DEFAULT_SLAB_PARAMS['num_slabs'],
                        help='Number of MIP slabs to produce along D. Output depth = num_slabs.')
    parser.add_argument('--slab_size', type=int, default=DEFAULT_SLAB_PARAMS['slab_size'],
                        help='Slab thickness (number of consecutive slices MIP-ed together).')
    parser.add_argument('--overlap', type=int, default=DEFAULT_SLAB_PARAMS['overlap'],
                        help='Slice overlap between consecutive slabs. Must be < slab_size. '
                             'Stride between slabs = slab_size - overlap.')
    cli_args = parser.parse_args()

    if cli_args.slabs and cli_args.overlap >= cli_args.slab_size:
        parser.error(f'--overlap ({cli_args.overlap}) must be < --slab_size ({cli_args.slab_size}).')

    cli_target_shape = (int(cli_args.height), int(cli_args.width), int(cli_args.depth))

    if cli_args.slabs:
        cli_slab_params = dict(
            num_slabs=int(cli_args.num_slabs),
            slab_size=int(cli_args.slab_size),
            overlap=int(cli_args.overlap),
        )
        if cli_args.out_dir is not None:
            cli_out_root = Path(cli_args.out_dir)
        else:
            cli_out_root = OUT_ROOT_BASE / (
                f"final_cropped_and_masked_slabs"
                f"_n{cli_slab_params['num_slabs']}"
                f"_s{cli_slab_params['slab_size']}"
                f"_o{cli_slab_params['overlap']}"
            )
    else:
        cli_slab_params = None
        if cli_args.out_dir is not None:
            cli_out_root = Path(cli_args.out_dir)
        elif cli_target_shape[2] != DEFAULT_TARGET_SHAPE[2]:
            cli_out_root = OUT_ROOT_BASE / f"final_cropped_and_masked_data_d{cli_target_shape[2]}"
        else:
            cli_out_root = OUT_ROOT  # unsuffixed default for backward compatibility

    main(target_shape=cli_target_shape, out_root=cli_out_root, slab_params=cli_slab_params)
