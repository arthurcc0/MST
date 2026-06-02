"""Inspect the depth distribution of new_penn source volumes.

Walks ``IN_DATA_ROOT`` (step2a's output) and reads the depth (last spatial
axis) of one reference NIfTI per case (default ``pre.nii.gz`` -- pre/post/sub
share the same shape for a given case). Prints a summary and writes a CSV
with one row per case.

Use this to decide slab parameters before running step2b in slab mode:

    python analyze_volume_depth.py
    python analyze_volume_depth.py -N 32 -S 3 -O 0

The script also reports how many cases would fall below the depth threshold
for the chosen (N, S, O) configuration, i.e. cases where step2b would have
to fall back to edge-slab repetition to reach N output slabs.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import nibabel as nib
from tqdm import tqdm


DEFAULT_IN_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4") / "preprocessed" / "data"
DEFAULT_FILE = "pre.nii.gz"


def collect_depths(in_root: Path, ref_filename: str) -> list:
    """Return list of dicts ``{patient_id, depth, ref_path}``."""
    if not in_root.is_dir():
        raise FileNotFoundError(f"Input data root not found: {in_root}")

    case_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    records = []
    for case in tqdm(case_dirs, desc="reading depths"):
        ref_path = case / ref_filename
        if not ref_path.exists():
            continue
        try:
            shape = nib.load(str(ref_path)).shape
        except Exception as e:
            print(f"  [warn] {case.name}: failed to read {ref_filename}: {e}", file=sys.stderr)
            continue
        if len(shape) < 3:
            continue
        records.append({"patient_id": case.name, "depth": int(shape[-1]), "ref_path": str(ref_path)})
    return records


def slab_required_depth(num_slabs: int, slab_size: int, overlap: int) -> int:
    """Min source D such that (N, S, O) slabs fit without edge-padding."""
    stride = slab_size - overlap
    if stride < 1:
        raise ValueError(f"overlap must be < slab_size (got S={slab_size}, O={overlap})")
    return (num_slabs - 1) * stride + slab_size


def print_summary(records: list, num_slabs: int, slab_size: int, overlap: int) -> None:
    if not records:
        print("No volumes found.")
        return
    depths = [r["depth"] for r in records]
    needed = slab_required_depth(num_slabs, slab_size, overlap)

    print(f"\nCases analyzed: {len(depths)}")
    print(f"Depth (slices): min={min(depths)}, median={median(depths):.1f}, "
          f"mean={mean(depths):.1f}, max={max(depths)}")
    print()
    print(f"Slab config: N={num_slabs}, S={slab_size}, O={overlap} "
          f"-> needs D >= {needed} slices to fit all {num_slabs} slabs natively.")
    below = sum(1 for d in depths if d < needed)
    too_shallow = sum(1 for d in depths if d < slab_size)
    print(f"  cases with D < {needed} (would fall back to edge-repetition): {below}  "
          f"({100*below/len(depths):.1f}%)")
    print(f"  cases with D < {slab_size} (cannot build a single slab):     {too_shallow}  "
          f"({100*too_shallow/len(depths):.1f}%)")

    # Compact distribution print: cap to 40 distinct depths.
    counter = Counter(depths)
    print("\nDepth distribution (count):")
    for d in sorted(counter):
        bar = "#" * min(60, counter[d])
        print(f"  D={d:>4d}: {counter[d]:>4d}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Print depth statistics for new_penn source volumes.")
    parser.add_argument("--in_root", default=str(DEFAULT_IN_ROOT),
                        help=f"Input data root (default: {DEFAULT_IN_ROOT}).")
    parser.add_argument("--ref_filename", default=DEFAULT_FILE,
                        help=f"Reference filename used to read the depth per case (default: {DEFAULT_FILE}).")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path. Default: <in_root>/../volume_depths.csv")
    parser.add_argument("-N", "--num_slabs", type=int, default=32, help="Slab config: number of slabs (for the summary).")
    parser.add_argument("-S", "--slab_size", type=int, default=3, help="Slab config: slab thickness (slices).")
    parser.add_argument("-O", "--overlap", type=int, default=0, help="Slab config: slice overlap.")
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_csv = Path(args.out_csv) if args.out_csv else (in_root.parent / "volume_depths.csv")

    records = collect_depths(in_root, args.ref_filename)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "depth", "ref_path"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {out_csv}")

    print_summary(records, args.num_slabs, args.slab_size, args.overlap)


if __name__ == "__main__":
    main()
