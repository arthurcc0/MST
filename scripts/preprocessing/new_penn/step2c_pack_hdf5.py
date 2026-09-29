"""Step 2c: pack a step2b output folder into one HDF5 file (``<folder>.h5``).

Pipeline:  step2b_apply_mask_split.py (nii.gz folder) -> this -> training on a cluster

The nii.gz folder stays the source of truth; this only repackages it so a
cluster job reads one file instead of ~13k small ones. ``PENN_Dataset3D``
switches to this file when ``path_root_data`` ends in ``.h5``.

Layout (N = packed exams, one chunk per exam)::

    uid          (N,)            str     exam id, e.g. "10009119_left"
    affine       (N, 4, 4)       float64 as stored in the NIfTI header
    <image>      (N, H, W, D)    float32 one per ``--images`` entry, e.g. "sub"
    skipped_uid  (M,)            str     exams left out (missing / constant volume)
    attrs: source_dir, images, created, git, n_packed, n_skipped

Volumes are read with NiBabel (not TorchIO/SimpleITK). Exams whose volume is
constant are skipped (same rule as ``PENN_Dataset3D._is_usable``). The file is
written as ``<out>.tmp`` and only renamed after ``--verify`` random exams read
back identical to their nii.gz.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
SRC_DIR = DATA_ROOT / "final_cropped_and_masked_slabs_n32_s3_o0"
# Only the subtraction is used by any planned experiment; add "pre", "post" to pack them too.
IMAGES = ("sub",)
NUM_WORKERS = 2


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path), mmap=False)
    data = np.ascontiguousarray(np.asanyarray(img.dataobj), dtype=np.float32)
    if data.ndim == 4:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[-1] == 1:
            data = data[..., 0]
        else:
            raise ValueError(f"{path}: expected a 3D volume, got {data.shape}")
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D, got {data.shape}")
    return np.ascontiguousarray(data, dtype=np.float32), np.asarray(img.affine, dtype=np.float64)


def _load_exam(src: str, uid: str, images: tuple[str, ...]):
    """Return (uid, {image: array}, affine) or (uid, None, reason)."""
    arrays, affine = {}, None
    for name in images:
        path = Path(src) / uid / f"{name}.nii.gz"
        if not path.is_file():
            return uid, None, f"missing {name}"
        try:
            arr, aff = _read(path)
        except Exception as e:
            return uid, None, f"unreadable {name}: {type(e).__name__}: {e}"
        if not arr.size or float(arr.max()) <= float(arr.min()):
            return uid, None, f"constant {name}"
        if affine is None:
            affine = aff
        elif not np.allclose(affine, aff):
            return uid, None, f"affine differs between images ({name})"
        arrays[name] = arr
    return uid, arrays, affine


def _iter_exams(src: str, uids: list[str], images: tuple[str, ...], num_workers: int):
    """Yield ``_load_exam`` results with at most ``num_workers`` volumes in flight."""
    if num_workers <= 1:
        for uid in uids:
            yield _load_exam(src, uid, images)
        return

    q: Queue = Queue(maxsize=num_workers)
    sentinel = object()

    def worker(subset: list[str]) -> None:
        try:
            for uid in subset:
                q.put(_load_exam(src, uid, images))
        finally:
            q.put(sentinel)

    threads = [
        Thread(target=worker, args=([uid for i, uid in enumerate(uids) if i % num_workers == w],), daemon=True)
        for w in range(num_workers)
    ]
    for t in threads:
        t.start()
    finished = 0
    while finished < num_workers:
        item = q.get()
        if item is sentinel:
            finished += 1
            continue
        yield item
    for t in threads:
        t.join()


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pack(src_dir: Path, out_path: Path, images=IMAGES, num_workers: int = NUM_WORKERS,
         compression: str = "gzip", level: int = 4, verify: int = 5, force: bool = False) -> Path:
    src_dir, out_path = Path(src_dir), Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists; pass --force to rebuild it.")
    uids = sorted(p.name for p in src_dir.iterdir() if p.is_dir())
    if not uids:
        raise FileNotFoundError(f"No exam folders under {src_dir}")
    images = tuple(images)
    print(f"Packing {len(uids)} exam folders from {src_dir}\n  images={images} -> {out_path}")

    for stale in (tmp, out_path):
        if stale.exists():
            stale.unlink()
            print(f"  removed leftover {stale.name}")

    comp = dict(compression=compression, compression_opts=level) if compression == "gzip" else (
        dict(compression=compression) if compression != "none" else {})
    packed, skipped = [], []
    with h5py.File(tmp, "w") as f:
        dsets, aff_ds, n = None, None, 0
        for uid, arrays, extra in tqdm(
            _iter_exams(str(src_dir), uids, images, num_workers), total=len(uids)
        ):
            if arrays is None:
                skipped.append((uid, extra))
                continue
            if dsets is None:
                shape = arrays[images[0]].shape
                dsets = {
                    name: f.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape),
                                           dtype="float32", chunks=(1, *shape), **comp)
                    for name in images
                }
                aff_ds = f.create_dataset("affine", shape=(0, 4, 4), maxshape=(None, 4, 4), dtype="float64")
            for name in images:
                if arrays[name].shape != dsets[name].shape[1:]:
                    raise ValueError(f"{uid}/{name}: shape {arrays[name].shape} != {dsets[name].shape[1:]}")
                dsets[name].resize(n + 1, axis=0)
                dsets[name][n] = arrays[name]
            aff_ds.resize(n + 1, axis=0)
            aff_ds[n] = extra
            packed.append(uid)
            n += 1
        str_dt = h5py.string_dtype()
        f.create_dataset("uid", data=np.array(packed, dtype=object), dtype=str_dt)
        f.create_dataset("skipped_uid", data=np.array([u for u, _ in skipped], dtype=object), dtype=str_dt)
        f.attrs.update(
            source_dir=str(src_dir), images=list(images), created=datetime.now().isoformat(timespec="seconds"),
            git=_git_rev(), n_packed=len(packed), n_skipped=len(skipped),
        )

    for uid, reason in skipped:
        print(f"  skipped {uid}: {reason}")
    print(f"Packed {len(packed)} exams, skipped {len(skipped)}.")

    check = random.Random(0).sample(packed, min(verify, len(packed)))
    with h5py.File(tmp, "r") as f:
        index = {u: i for i, u in enumerate(f["uid"].asstr()[()])}
        for uid in check:
            for name in images:
                arr, aff = _read(src_dir / uid / f"{name}.nii.gz")
                i = index[uid]
                if not (np.array_equal(f[name][i], arr) and np.array_equal(f["affine"][i], aff)):
                    raise RuntimeError(f"Verification failed for {uid}/{name}; keeping {tmp} for inspection.")
    print(f"Verified {len(check)} random exams against their nii.gz.")

    os.replace(tmp, out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=SRC_DIR, help="step2b output folder.")
    parser.add_argument("--out", type=Path, default=None, help="Default: <src>.h5 next to the folder.")
    parser.add_argument("--images", nargs="+", default=list(IMAGES), help="Image names to pack (default: sub).")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Reader threads (default: 2). Use 1 if RAM is tight.")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--level", type=int, default=4, help="gzip level (default: 4).")
    parser.add_argument("--verify", type=int, default=5, help="Random exams to re-check (default: 5).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file (and leftover .tmp).")
    args = parser.parse_args()
    out = args.out or args.src.with_name(args.src.name + ".h5")
    pack(args.src, out, args.images, args.num_workers, args.compression, args.level, args.verify, args.force)


if __name__ == "__main__":
    main()
