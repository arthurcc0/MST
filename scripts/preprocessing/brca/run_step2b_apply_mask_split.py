"""Run new_penn step2b slab pipeline with Basser/BRCA paths.

Always writes to ``config.FINAL_SLAB_DIR`` under ``config.OUT_ROOT``.
Do **not** run ``scripts/preprocessing/new_penn/step2b_apply_mask_split.py``
directly for this cohort — that defaults to ``MST_birads4`` paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_PENN_DIR = SCRIPT_DIR.parent / "new_penn"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(NEW_PENN_DIR))

import step2b_apply_mask_split as step2b  # noqa: E402
from config import DEFAULT_DEPTH, FINAL_SLAB_DIR, MASK_DIR, OUT_ROOT  # noqa: E402

# Patch module-level paths before any workers spawn.
step2b.OUT_ROOT_BASE = OUT_ROOT
step2b.IN_DATA_ROOT = OUT_ROOT / "preprocessed" / "data"
step2b.MASK_DIR = MASK_DIR
step2b.OUT_ROOT = FINAL_SLAB_DIR
step2b.TARGET_SHAPE = (224, 224, DEFAULT_DEPTH)
step2b.DEFAULT_SLAB_PARAMS = dict(num_slabs=32, slab_size=3, overlap=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BRCA wrapper: mask + split L/R + slab MIP to FINAL_SLAB_DIR."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FINAL_SLAB_DIR,
        help=f"Slab output root (default: {FINAL_SLAB_DIR}).",
    )
    parser.add_argument("--num-slabs", type=int, default=32)
    parser.add_argument("--slab-size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=step2b.NUM_WORKERS)
    parser.add_argument("--pool-chunksize", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mask-cc-min-fraction",
        type=float,
        default=step2b.MASK_CC_MIN_FRACTION,
    )
    args = parser.parse_args()

    if args.overlap >= args.slab_size:
        parser.error(f"--overlap ({args.overlap}) must be < --slab-size ({args.slab_size}).")

    out_dir = Path(args.out_dir)
    slab_params = dict(
        num_slabs=int(args.num_slabs),
        slab_size=int(args.slab_size),
        overlap=int(args.overlap),
    )

    print(f"[BRCA step2b] input:  {step2b.IN_DATA_ROOT}")
    print(f"[BRCA step2b] output: {out_dir}")
    print(f"[BRCA step2b] masks:  {step2b.MASK_DIR}")

    step2b.main(
        target_shape=(224, 224, DEFAULT_DEPTH),
        out_root=out_dir,
        slab_params=slab_params,
        num_workers=int(args.num_workers),
        pool_chunksize=int(args.pool_chunksize),
        force=bool(args.force),
        mask_cc_min_fraction=float(args.mask_cc_min_fraction),
    )


if __name__ == "__main__":
    main()
