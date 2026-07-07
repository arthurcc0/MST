"""Compare slice/plane attention between checkpoints (one test sample)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mst.data.datasets.dataset_3d_penn import PENN_Dataset3D
from mst.models.dino import DinoClassifierSlice


def inspect_run(run_dir: Path, fold: int, split_csv: str, data_root: str):
    df = PENN_Dataset3D.load_split(split_csv, fold=fold, split="test")
    ds = PENN_Dataset3D(
        df=df.head(1),
        path_root_data=data_root,
        image_crop=(224, 224, 32),
    )
    batch = ds[0]
    source = batch["source"].unsqueeze(0).float()
    uid = batch["uid"]
    print(f"\n=== {run_dir.name} | UID {uid} ===")

    model = DinoClassifierSlice.load_best_checkpoint(run_dir, use_registers=True)
    model.eval()
    with torch.no_grad():
        pred = model(source, save_attn=True)
        n_slice_maps = len(model.attention_maps_slice)
        n_plane_maps = len(model.attention_maps)
        print(f"  hooked slice maps: {n_slice_maps}, plane maps: {n_plane_maps}")
        if n_slice_maps:
            raw = model.attention_maps_slice[-1]
            cls_to_slices = raw[:, :, 0, 1:]
            print(
                f"  raw slice attn CLS->slices: sum={float(cls_to_slices.sum()):.6f} "
                f"max={float(cls_to_slices.max()):.6f} shape={tuple(raw.shape)}"
            )
            print(
                f"  CLS self-attn: {float(raw[:, :, 0, 0].mean()):.6f}"
            )
        plane = model.get_plane_attention()
        slice_a = model.get_slice_attention()
        prod = model.get_attention_maps()
        print(
            f"  get_plane_attention sum={float(plane.sum()):.4f} max={float(plane.max()):.6f}"
        )
        print(
            f"  get_slice_attention sum={float(slice_a.sum()):.6f} max={float(slice_a.max()):.6f}"
        )
        print(
            f"  get_attention_maps sum={float(prod.sum()):.6f} max={float(prod.max()):.6f}"
        )
        print(f"  pred logits: {pred.detach().cpu().numpy().ravel()}")


def scan_run(run_name: str, fold: int, split_csv: str, n: int = 10):
    run = ROOT / "runs/PENN" / run_name
    df = PENN_Dataset3D.load_split(split_csv, fold=fold, split="test")
    ds = PENN_Dataset3D(
        df=df.head(n),
        path_root_data=r"D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0",
        image_crop=(224, 224, 32),
    )
    print(f"\n=== {run_name} scan ({n} cases) ===")
    model = DinoClassifierSlice.load_best_checkpoint(run, use_registers=True)
    model.eval()
    zero_head_counts = []
    for i in range(len(ds)):
        batch = ds[i]
        source = batch["source"].unsqueeze(0).float()
        with torch.no_grad():
            model(source, save_attn=True)
        raw = model.attention_maps_slice[-1]
        cls_slices = raw[:, :, 0, 1:]
        per_head = cls_slices.sum(dim=-1).squeeze()
        zero_heads = int((per_head < 1e-8).sum())
        zero_head_counts.append(zero_heads)
        prod = model.get_attention_maps()
        prod_ok = not torch.isnan(prod).any()
        print(
            f"  UID {batch['uid']}: zero_heads={zero_heads}/12 "
            f"self_mean={float(raw[:, :, 0, 0].mean()):.4f} prod_ok={prod_ok}"
        )
    print(f"  mean zero_heads: {sum(zero_head_counts)/len(zero_head_counts):.1f}")


def scan_v5_ce():
    scan_run(
        "DinoClassifierSlice_2026_06_16_001131_subtraction_birads4_pennPreTr123_frozebckbn_headonly_notBalancedCE_v5_lr1x10-6_f1_reg",
        1,
        r"D:\Users\arthur\Data\MST_birads4\new_penn_datasplit_v3.csv",
    )
    scan_run(
        "DinoClassifierSlice_2026_06_02_114802_subtraction_new_penn_from_dino_freezebckbone_headonly_notBalancedCE_f0_reg_multi",
        0,
        r"D:\Users\arthur\Data\MST_birads4\new_penn_datasplit_v2.csv",
    )


if __name__ == "__main__":
    runs = ROOT / "runs" / "PENN"
    data = r"D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
    cases = [
        (
            runs / "DinoClassifierSlice_2026_05_05_221557_subtraction_reg_multi",
            0,
            r"D:\Users\arthur\Data\MST_birads4\old_penn_datasplit_v2.csv",
        ),
        (
            runs / "DinoClassifierSlice_2026_06_09_065654_subtraction_old_penn_balancedCE_seed123_f0_reg_multi",
            0,
            r"D:\Users\arthur\Data\MST_birads4\old_penn_datasplit_v2.csv",
        ),
        (
            runs / "DinoClassifierSlice_2026_06_02_114802_subtraction_new_penn_from_dino_freezebckbone_headonly_notBalancedCE_f0_reg_multi",
            0,
            r"D:\Users\arthur\Data\MST_birads4\new_penn_datasplit_v2.csv",
        ),
        (
            runs / "DinoClassifierSlice_2026_06_16_001131_subtraction_birads4_pennPreTr123_frozebckbn_headonly_notBalancedCE_v5_lr1x10-6_f1_reg",
            1,
            r"D:\Users\arthur\Data\MST_birads4\new_penn_datasplit_v3.csv",
        ),
        (
            runs / "DinoClassifierSlice_2026_06_16_085216_subtraction_birads4_focal_g2_balanced_headonly_lr1e-6_v5_f1_reg",
            1,
            r"D:\Users\arthur\Data\MST_birads4\new_penn_datasplit_v3.csv",
        ),
    ]
    for run_dir, fold, split_csv in cases:
        if not run_dir.exists():
            print(f"SKIP missing {run_dir}")
            continue
        inspect_run(run_dir, fold, split_csv, data)
    scan_v5_ce()
