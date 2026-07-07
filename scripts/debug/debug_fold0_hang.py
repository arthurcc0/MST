"""Time fold-0 first test case through predict path (find hang)."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.predict.main_predict import run_pred

from mst.data.datasets.dataset_3d_penn import PENN_Dataset3D
from mst.models.dino import DinoClassifierSlice
from mst.models.utils.functions import minmax_norm, tensor2image, tensor_cam2image
from torchvision.utils import save_image

DATA = r"D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
SPLIT = r"D:\Users\arthur\Data\MST_birads4\new_penn_datasplit_v3.csv"
RUN = ROOT / "runs/PENN/DinoClassifierSlice_2026_06_16_083352_subtraction_birads4_focal_g2_balanced_headonly_lr1e-6_v5_f0_reg"
OUT = ROOT / "results-pretrained-oldpenn-on-newpenn/PENN/_debug_fold0_save"


def tick(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"  [{now - t0:6.1f}s] {label}")
    return now


def main():
    t0 = time.perf_counter()
    df = PENN_Dataset3D.load_split(SPLIT, fold=0, split="test")
    print(f"fold 0 test n={len(df)}, first UID={df.UID.iloc[0]}")
    ds = PENN_Dataset3D(df=df.head(1), path_root_data=DATA, image_crop=(224, 224, 32))
    batch = ds[0]
    batch["source"] = batch["source"].unsqueeze(0).float()
    t0 = tick(f"__getitem__ UID={batch['uid']}", t0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoClassifierSlice.load_best_checkpoint(RUN, use_registers=True)
    model.to(device).eval()
    t0 = tick("model loaded", t0)

    batch["source"] = batch["source"].to(device)
    source = batch["source"]
    _, weight, weight_slice = run_pred(model, batch, save_attn=True, use_softmax=False)
    t0 = tick("run_pred", t0)

    OUT.mkdir(parents=True, exist_ok=True)
    weight_slice = weight_slice.detach().cpu()
    weight_slice /= weight_slice.sum()
    weight = weight.detach().cpu()
    weight = weight.clip(*__import__("numpy").quantile(weight, [0.995, 0.999]))
    save_image(tensor2image(source.rot90(2, (2, 3))), str(OUT / "test_input.png"), normalize=True)
    t0 = tick("save input png", t0)
    source_for_viz = source.mean(dim=1, keepdim=True)
    save_image(
        tensor_cam2image(
            minmax_norm(source_for_viz.rot90(2, (2, 3))),
            minmax_norm(weight.rot90(2, (2, 3))),
            alpha=0.5,
        ),
        str(OUT / "test_overlay.png"),
        normalize=False,
    )
    t0 = tick("save overlay png", t0)


if __name__ == "__main__":
    main()
