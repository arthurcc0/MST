import sys
import os
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
# Get the directory containing the script (e.g., /path/to/MST/scripts)
script_dir = os.path.dirname(script_path)
# Get the project root directory (e.g., /path/to/MST)
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pathlib import Path
import argparse
import logging
import yaml
from tqdm import tqdm
import math 
import torch 
import torch.nn as nn
import torchio as tio
import numpy as np 
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch.nn.functional as F
import pandas as pd 
from torchvision.utils import save_image
from monai.metrics import compute_average_surface_distance, compute_iou, DiceMetric, compute_dice

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_penn import PENN_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoClassifierSlice
from mst.models.dino_v3 import DinoClassifierSliceV3
from mst.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from mst.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot
from mst.utils.random_seed import set_random_seed

from aggregate_auc import results_dir_name

set_random_seed(42)


def _resolve_run_path(run_dir: str | Path, run_folder: str | Path) -> Path:
    run_folder = Path(run_folder)
    if run_folder.is_absolute():
        return run_folder
    return Path(run_dir) / run_folder


def _load_train_config(path_run: Path) -> dict:
    config_file = path_run / 'config.yaml'
    if not config_file.is_file():
        return {}
    with open(config_file, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def _slices_from_run_folder_name(run_folder_name: str) -> int | None:
    for token in run_folder_name.split('_'):
        if token.startswith('d') and len(token) > 1 and token[1:].isdigit():
            return int(token[1:])
    return None


def _split_results_stem(split: str) -> str:
    """Suffix for split-specific predict artifacts (train/val/test)."""
    return split if split in {'train', 'val', 'test'} else ''


def _normalize_class_label(label) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ""
    text = str(label).strip().lower().replace("-", " ")
    aliases = {
        "benign": "benign",
        "high risk": "highrisk",
        "highrisk": "highrisk",
        "malignant": "malignant",
    }
    if text in aliases:
        return aliases[text]
    compact = text.replace(" ", "")
    if compact in {"benign", "highrisk", "malignant"}:
        return compact
    return compact or "unknown"


def _class_label_from_batch(batch, gt_label: int) -> str:
    for key in ("label", "class_label", "mapping_label", "split_label"):
        if key not in batch:
            continue
        raw = batch[key]
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        elif torch.is_tensor(raw):
            raw = raw.item() if raw.numel() == 1 else raw[0].item()
        label = _normalize_class_label(raw)
        if label:
            return label
    return "malignant" if gt_label == 1 else "benign"


def _attention_filename_stem(uid: str, class_label: str, score: float) -> str:
    safe_uid = str(uid).replace("\\", "_").replace("/", "_").replace(":", "_")
    return f"{safe_uid}_{class_label}_{score:.4f}"


def _normalize_slice_weight_volume(weight_slice: torch.Tensor) -> torch.Tensor:
    ws = weight_slice.detach().float().cpu()
    total = ws.sum()
    if float(total) > 1e-8 and not torch.isnan(total):
        return ws / total
    return torch.ones_like(ws) / ws.numel()


def _clip_cam_weight(weight: torch.Tensor) -> torch.Tensor:
    w = weight.detach().float().cpu()
    if float(w.abs().sum()) < 1e-8 or torch.isnan(w).any():
        return w
    lo, hi = np.quantile(w.numpy(), [0.995, 0.999])
    return w.clip(lo, hi)


def _dino_attention_volumes(model, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plane-only, slice-only, and product volumes at patch resolution [1, 1, D, h, w]."""
    patch = getattr(model, "patch_size", None)
    if patch is None and hasattr(model, "encoder"):
        patch = getattr(model.encoder, "patch_size", 14)
    patch = int(patch or 14)
    d, h, w = source.shape[2], source.shape[3] // patch, source.shape[4] // patch

    weight_plane = model.get_plane_attention().mean(dim=1).view(1, 1, d, h, w)

    raw = model.attention_maps_slice[-1][:, :, 0, 1:]
    normed = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    slice_1d = normed.mean(dim=1)
    weight_slice = slice_1d.view(1, 1, -1, 1, 1)

    weight_product = weight_plane * slice_1d.view(1, 1, d, 1, 1)
    return weight_plane, weight_slice, weight_product


def _save_attention_overlay(
    source: torch.Tensor,
    weight: torch.Tensor,
    path: Path,
    normalize_slice: bool = False,
) -> None:
    """Save MRI + CAM overlay; weight is trilinear-sized to match source."""
    if normalize_slice:
        weight = _normalize_slice_weight_volume(weight)
    else:
        weight = _clip_cam_weight(weight)
    source_for_viz = source.mean(dim=1, keepdim=True)
    save_image(
        tensor_cam2image(
            minmax_norm(source_for_viz.rot90(2, (2, 3))),
            minmax_norm(weight.rot90(2, (2, 3))),
            alpha=0.5,
        ),
        str(path),
        normalize=False,
    )


def get_dataset(name, split, run_folder, **kwargs):
    if name == 'DUKE':
        # Filter arguments for DUKE_Dataset3D constructor (similar to main_train.py)
        duke_kwargs = {
            'path_root': kwargs.get('path_root', None),
            'fold': kwargs.get('fold', 0),
            'split': split,
            'fraction': kwargs.get('fraction', None),
            'transform': None,
            'image_resize': kwargs.get('image_resize', None),
            'resample': kwargs.get('resample', None),
            'flip': kwargs.get('flip', False),
            'random_rotate': kwargs.get('random_rotate', False),
            'image_crop': kwargs.get('image_crop', (224, 224, 32)),
            'random_center': kwargs.get('random_center', False),
            'noise': kwargs.get('noise', False),
            'to_tensor': True,
            'get_segmentation': kwargs.get('get_segmentation', False)
        }
        return DUKE_Dataset3D(**duke_kwargs)
    elif name == 'LIDC':
        return LIDC_Dataset3D(split=split, **kwargs)
    elif name == 'MRNet':
        return MRNet_Dataset3D(split=split, **kwargs)
    elif name == 'PENN':
        penn_split_csv = kwargs.pop('penn_split_csv', None)
        fold = int(kwargs.pop('fold', 0))
        path_csv = PENN_Dataset3D.resolve_split_csv_path(penn_split_csv)
        print(f"[PENN] split CSV: {path_csv} (fold={fold}, split={split!r})")
        df_split = PENN_Dataset3D.load_split(path_csv, fold=fold, split=split)
        return PENN_Dataset3D(df=df_split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_model(name, **kwargs):
    if name == 'ResNet':
        return ResNet
    elif name == 'ResNetSliceTrans':
        return ResNetSliceTrans
    elif name in ('DinoClassifierSlice', 'DinoV2ClassifierSlice'):
        return DinoClassifierSlice
    elif name == 'DinoClassifierSliceV3':
        return DinoClassifierSliceV3
    else:
        raise ValueError(f"Unknown model: {name}")


def _pred_trans(
    model,
    source,
    src_key_padding_mask,
    save_attn=False,
    use_softmax=True,
    model_forward_kwargs=None,
):
    model_forward_kwargs = model_forward_kwargs or {}
    # Run model
    if isinstance(model, ResNetSliceTrans) and save_attn:
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn, **model_forward_kwargs)
    else:
        with torch.no_grad():
            pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn, **model_forward_kwargs)

    if use_softmax: # Necessary to standardize the scale before TTA average 
        pred = torch.softmax(pred, dim=-1)

    if not save_attn:
        return pred, None, None, None

    if isinstance(model, (DinoClassifierSlice, DinoClassifierSliceV3)):
        weight_plane, weight_slice, weight = _dino_attention_volumes(model, source)
        weight_slice = weight_slice * torch.ones_like(source, device=weight.device)
    else:
        # Spatial attention (product path for ResNet slice models)
        weight = model.get_attention_maps()  # [B*D, Heads, HW]
        weight = weight.mean(dim=1) # Mean of heads
        spatial_shape = weight.shape[-2:] if isinstance(model, ResNetSliceTrans) else torch.tensor(source.shape[3:]) // 14
        weight = weight.view(1, 1, source.shape[2], *spatial_shape)
        weight_plane = weight.clone()
        weight_slice = model.get_slice_attention() # [B*D, Heads, 1]
        weight_slice = weight_slice.mean(dim=1) # Mean of heads
        weight_slice = weight_slice.view(1, 1, -1, 1, 1) * torch.ones_like(source, device=weight.device)

    return pred, weight, weight_slice, weight_plane




def _pred_resnet(model, source, src_key_padding_mask, save_attn=False, use_softmax=True):
    # Run model
    if save_attn: # Grads required 
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=True)
    else:
        with torch.no_grad():
            pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=False)
    
    if use_softmax: # Necessary to standardize the scale before TTA average 
        pred = torch.softmax(pred, dim=-1)
      
    if not save_attn:
        return pred, None, None, None

    weight = model.get_attention_maps()

    # Slice attention (dummy)
    weight_slice = torch.ones_like(source, device=weight.device)

    return pred, weight, weight_slice, None


def run_pred(
    model,
    batch,
    save_attn=False,
    use_softmax=True,
    use_tta=False,
    embeddings_only=False,
):
    source, src_key_padding_mask = batch['source'], batch.get('src_key_padding_mask', None)
    pred_func = None 
    if isinstance(model, ResNetSliceTrans): 
        pred_func = _pred_trans
    elif isinstance(model, ResNet):
        pred_func = _pred_resnet
    elif isinstance(model, (DinoClassifierSlice, DinoClassifierSliceV3)):
        pred_func = _pred_trans

    model_forward_kwargs = {}
    if embeddings_only and isinstance(model, (DinoClassifierSlice, DinoClassifierSliceV3)):
        model_forward_kwargs['without_linear'] = True
        use_softmax = False

    pred, weight, weight_slice, weight_plane = pred_func(
        model,
        source,
        src_key_padding_mask,
        save_attn=save_attn,
        use_softmax=use_softmax,
        model_forward_kwargs=model_forward_kwargs,
    )

    if use_tta:
        for flip_dim in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4),]:
            pred_i, weight_i, weight_slice_i, weight_plane_i = pred_func(
                model,
                torch.flip(source, flip_dim),
                src_key_padding_mask,
                save_attn=save_attn,
                use_softmax=use_softmax,
                model_forward_kwargs=model_forward_kwargs,
            )
            pred = pred + pred_i
            if save_attn:
                weight = weight + torch.flip(weight_i, flip_dim)
                weight_slice = weight_slice + torch.flip(weight_slice_i, flip_dim)
                if weight_plane is not None and weight_plane_i is not None:
                    weight_plane = weight_plane + torch.flip(weight_plane_i, flip_dim)

        pred = pred / 8
        if save_attn:
            weight = weight / 8
            weight_slice = weight_slice / 8
            if weight_plane is not None:
                weight_plane = weight_plane / 8

    # Interpolate to required size 
    if save_attn:
        size = source.shape[2:]
        weight = F.interpolate(weight, size=size, mode='trilinear')
        if weight_plane is not None:
            weight_plane = F.interpolate(weight_plane, size=size, mode='trilinear')

    return pred, weight, weight_slice, weight_plane 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default='./runs', type=str)
    parser.add_argument('--run_folder', default='LIDC/ResNet', type=str)
    parser.add_argument(
        '--penn_split_csv',
        default=None,
        type=str,
        help=(
            'PENN splits CSV (Fold, Split, UID, Malignant, …). '
            'Filename under PENN_Dataset3D.AUX_PATH or absolute path. '
            'Default: new_penn_datasplit.csv; overridden by config.yaml in the run folder when omitted.'
        ),
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='PENN cross-validation fold. Default: fold from config.yaml in the run folder, else 0.',
    )
    parser.add_argument('--use_registers', type=lambda x: str(x).lower() == 'true', nargs='?', const=True, default=False,help='Load DINOv2 backbone with 4 register tokens (Darcet et al., 2023).')
    parser.add_argument('--slices', type=int, default=None,
                        help='Number of slices D for image_crop=(224, 224, D). If omitted, auto-detect from a "_d{N}" tag in the run folder name (default 32).')
    parser.add_argument('--path_root_data', type=str, default=None,
                        help="Override for the preprocessed-data folder (PENN). Accepts a bare folder name "
                             "resolved under PENN_Dataset3D.AUX_PATH (e.g. 'final_cropped_and_masked_data_d64') "
                             "or an absolute path. Defaults to 'final_cropped_and_masked_data'.")
    parser.add_argument('--slab_tissue_soft_weight', action='store_true')
    parser.add_argument('--slab_tissue_min_weight', type=float, default=0.1)
    parser.add_argument('--slab_tissue_keep_ratio', type=float, default=0.5)
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test', 'all'],
        help=(
            'PENN split to evaluate (train / val / test / all). '
            'With --save_embeddings, run once per split (train, val, test) into the same '
            'embeddings_fused/ folder before embedding_mlp.py --eval_mode penn_split.'
        ),
    )
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument(
        '--results-name',
        type=str,
        default=None,
        help=(
            'Short directory name under results/<dataset>/ for outputs. '
            'Default: cohort tag after subtraction_ in the run folder name '
            '(e.g. birads4_..._f2_reg).'
        ),
    )
    parser.add_argument('--get_attention', action='store_true', help='Flag to get attention')
    parser.add_argument(
        '--attention_malignant_only',
        action='store_true',
        help='With --get_attention, save attention overlays only for ground-truth malignant cases (target==1).',
    )
    parser.add_argument('--get_segmentation', action='store_true', help='Flag to get attention')
    parser.add_argument('--use_tta', action='store_true', help='Use test time augmentation')
    parser.add_argument(
        '--save_embeddings',
        action='store_true',
        help=(
            'Save fused per-volume CLS embeddings (ViT+DINO slice fusion output, '
            'before the classification head). Written as <UID>_pred.npz with key '
            '\'pred\' (float32 shape [batch, embed_dim]).'
        ),
    )

    args = parser.parse_args()
    get_attention = args.get_attention
    attention_malignant_only = args.attention_malignant_only
    get_segmentation = args.get_segmentation
    use_tta = args.use_tta
    save_embeddings = args.save_embeddings
    print(f"Using TTA {use_tta}")
    print(f"Save embeddings {save_embeddings}")
    if attention_malignant_only and not get_attention:
        print("Warning: --attention_malignant_only has no effect without --get_attention.")
    elif attention_malignant_only:
        print("Attention maps: malignant cases only (GT==1).")

    path_run = _resolve_run_path(args.run_dir, args.run_folder)
    run_folder = path_run
    dataset = run_folder.parent.name
    logger = logging.getLogger(__name__)
    train_cfg = _load_train_config(path_run)
    model_name = train_cfg.get('model_name') or run_folder.name.split('_', 1)[0]

    # Resolve slice count: CLI > config.yaml > _d{N} in run folder name > 32
    if args.slices is not None:
        slices = int(args.slices)
    elif train_cfg.get('slices') is not None:
        slices = int(train_cfg['slices'])
    else:
        slices = _slices_from_run_folder_name(run_folder.name) or 32
    image_crop = (224, 224, slices)

    fold = int(args.fold) if args.fold is not None else int(train_cfg.get('fold', 0))
    path_root_data = args.path_root_data or train_cfg.get('path_root_data')
    penn_split_csv = (
        args.penn_split_csv
        or train_cfg.get('penn_split_csv')
        or train_cfg.get('split_csv_path')
    )
    penn_split_resolved = None
    if dataset == 'PENN':
        penn_split_resolved = PENN_Dataset3D.resolve_split_csv_path(penn_split_csv)
        print(f"PENN split CSV: {penn_split_resolved}")
        print(f"PENN fold: {fold}")
        if path_root_data:
            print(f"PENN path_root_data: {path_root_data}")
        if train_cfg.get('cohort'):
            print(f"PENN training cohort tag: {train_cfg['cohort']}")
    if 'DinoV2ClassifierSlice_2025_08_25_190331_multi' in str(run_folder):
        logger.info(
            f"Overriding model_name from '{model_name}' to 'DinoClassifierSlice' "
            f"(legacy run folder) for run: {run_folder}"
        )
        model_name = "DinoClassifierSlice"
    #------------ Settings/Defaults ----------------
    results_folder = 'results_tta'if use_tta else 'results-pretrained-oldpenn-on-newpenn'
    results_name = args.results_name or results_dir_name(path_run, fold=fold)
    path_out = Path(args.output_dir) / results_folder / path_run.parent.name / results_name
    path_out.mkdir(parents=True, exist_ok=True)
    (path_out / "source_run.txt").write_text(str(path_run.resolve()), encoding="utf-8")
    print(f"Predict outputs: {path_out.resolve()}")
    if results_name != path_run.name:
        print(f"  (training run: {path_run.resolve()})")
    split_stem = _split_results_stem(args.split)
    results_csv_name = f'results_{split_stem}.csv' if split_stem else 'results.csv'
    roc_png_name = f'roc_{split_stem}.png' if split_stem else 'roc.png'
    cm_png_name = f'confusion_matrix_{split_stem}.png' if split_stem else 'confusion_matrix.png'
    device = torch.device("cuda")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}
    torch.set_float32_matmul_precision('high')

    # ------------ Logging --------------------
    logger.setLevel(logging.INFO) 
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(path_out / f'{Path(__file__).name}.txt', mode='w'))

    # ------------ Load Data ----------------
    penn_eval_split = None if args.split == 'all' else args.split
    ds_kwargs = dict(
        name=dataset,
        split=penn_eval_split,
        run_folder=str(path_run),
        get_segmentation=get_segmentation,
        penn_split_csv=str(penn_split_resolved) if penn_split_resolved is not None else None,
        fold=fold,
        image_crop=image_crop,
    )
    if path_root_data is not None:
        ds_kwargs['path_root_data'] = path_root_data
    if dataset == 'PENN':
        ds_kwargs.update({
            'slab_tissue_soft_weight': args.slab_tissue_soft_weight,
            'slab_tissue_min_weight': args.slab_tissue_min_weight,
            'slab_tissue_keep_ratio': args.slab_tissue_keep_ratio,
        })
    ds_test = get_dataset(**ds_kwargs)
    logger.info(f"Using image_crop={image_crop} (slices={slices}), PENN split={args.split!r}.")

    dm = DataModule(
        ds_test=ds_test,
        batch_size=1, 
        num_workers=0,
        pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    # Auto-detect '_reg' tag from the run folder name (set by main_train.py when
    # --use_registers is on). CLI flag still wins if explicitly passed.
    use_registers = args.use_registers or ('reg' in run_folder.name.split('_'))

    model_kwargs = {}
    if model_name in ('DinoClassifierSlice', 'DinoV2ClassifierSlice'):
        model_kwargs['use_registers'] = use_registers
        if use_registers:
            logger.info("Using DINOv2 with registers backbone (detected via run folder name or --use_registers).")
    elif model_name == 'DinoClassifierSliceV3':
        logger.info("Using DINOv3 ViT backbone (4 register tokens, patch size 16).")

    model = get_model(model_name).load_best_checkpoint(path_run, **model_kwargs)
    # model = get_model(model_name).load_best_checkpoint(path_run)
    # # Legacy path extracts penultimate embeddings by replacing the head with Identity.
    # if not save_embeddings:
    #     model.linear = nn.Identity()

    model.to(device)
    model.eval()

    classification = []
    results = []
    results_seg = []
    counter = 0 
    embedding_rows = []
    n_cases = len(ds_test)
    if get_attention:
        logger.info(
            "Generating attention for %d test cases. The progress bar stays at 0%% until the "
            "first case finishes (often 1–3 min on a cold GPU: NIfTI load + CUDA warmup + forward + PNGs).",
            n_cases,
        )
        print(
            f"Attention: {n_cases} cases on fold {fold}. "
            "Bar may sit at 0% for 1–3 min before showing 1/{n_cases}.",
            flush=True,
        )

    test_loader = dm.test_dataloader()
    progress = tqdm(test_loader, mininterval=1.0 if get_attention else 0.1)
    for n, batch in enumerate(progress):
        logger.debug(f"Processing batch {n}, Keys: {list(batch.keys())}, Target: {batch.get('target', 'N/A')}, UID: {batch.get('uid', 'N/A')}")
        batch['source'] = batch['source'].to(device).float()
        source, target = batch['source'], batch['target']
        uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())

        if save_embeddings:
            path_emb = path_out / 'embeddings_fused'
            path_emb.mkdir(parents=True, exist_ok=True)
            pred_emb, _, _, _ = run_pred(
                model,
                batch,
                save_attn=False,
                use_softmax=False,
                use_tta=use_tta,
                embeddings_only=True,
            )
            vec = pred_emb.detach().float().cpu().numpy().astype(np.float32)
            safe_uid = str(uid).replace('\\', '_').replace('/', '_').replace(':', '_')
            out_npz = path_emb / f'{safe_uid}_pred.npz'
            np.savez_compressed(out_npz, pred=vec)
            gt_val = target.squeeze().detach().cpu()
            embedding_rows.append({
                'UID': uid,
                'embedding_dim': int(vec.shape[-1]),
                'path': str(out_npz.resolve()),
                'GT': int(gt_val.item()) if gt_val.numel() == 1 else '',
            })
            continue

        counter = 0
        if get_segmentation:
            # Skip cases without target label 
            # if target != 1:
            #     continue 
            # counter += 1
            # if counter > 5:
            #    break
            # Skip cases without at least two raters
            if 'mask_1' not in batch:
                logger.info(f"Excluding UID: {uid}")
                continue

            # Run prediction 
            pred, weight, weight_slice, _ = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)
            
            # Transfer weights to binary segmentation mask   
            weight = weight.detach().cpu()
            seg = (weight>np.quantile(weight, 0.999)).type(torch.int16)
            seg_hot = one_hot(seg[:, 0], 2)

            seg_gt = batch['mask']
            seg_hot_gt = one_hot(batch['mask'][:,0])
            spacing = batch['affine'][0].diag()[:3]
            vol = math.prod(spacing)

            dice = compute_dice(y_pred=seg_hot, y=seg_gt, include_background=True)
            iou = compute_iou(y_pred=seg_hot, y=seg_hot_gt, include_background=True)
            assd = compute_average_surface_distance(y_pred=seg_hot, y=seg_hot_gt, 
                    include_background=True, symmetric=True, spacing=spacing.tolist())


            results_seg.append({
                'UID': uid,
                'Path': batch['path'][0],
                'Voxel':seg_gt.sum().item(),
                'Volume': (seg_gt.sum()*vol).item(),
                'Dice':dice.mean().item(),
                'IOU':iou.mean().item(),
                'ASSD': assd.mean().item(),
                'Dice_foreground':dice[0, 1].item(),
                'IOU_foreground':iou[0, 1].item(),
                'ASSD_foreground': assd[0, 1].item(),
            })



        elif get_attention:
            gt_label = int(target.squeeze().detach().cpu().item())
            save_attn_maps = not (attention_malignant_only and gt_label != 1)

            path_out_dir = path_out / 'attention-rotated'
            path_out_dir.mkdir(parents=True, exist_ok=True)

            pred, weight, weight_slice, weight_plane = run_pred(
                model, batch, save_attn=save_attn_maps, use_softmax=False, use_tta=use_tta
            )
            pred = pred.detach().cpu()
            pred_binary = torch.argmax(pred, dim=1)
            pred_prob = torch.softmax(pred, dim=-1)[:, 1]
            row = {
                'UID': uid,
                'GT': gt_label,
                'NN': pred_binary.item(),
                'NN_pred': pred_prob.item(),
            }
            if save_attn_maps and weight is not None:
                row['attn_plane_max'] = float(weight_plane.detach().max().cpu()) if weight_plane is not None else ''
                row['attn_slice_sum'] = float(weight_slice.detach().sum().cpu())
                row['attn_product_max'] = float(weight.detach().max().cpu())
            classification.append(row)

            if not save_attn_maps:
                logger.debug(f"Skipping attention maps for benign UID {uid} (GT={gt_label}).")
            else:
                class_label = _class_label_from_batch(batch, gt_label)
                score = float(pred_prob.item())
                file_stem = _attention_filename_stem(uid, class_label, score)
                try:
                    save_image(
                        tensor2image(source.rot90(2, (2, 3))),
                        str(path_out_dir / f'{file_stem}_input.png'),
                        normalize=True,
                    )
                    if weight_plane is not None:
                        _save_attention_overlay(
                            source, weight_plane, path_out_dir / f'{file_stem}_overlay_plane.png',
                        )
                    _save_attention_overlay(
                        source,
                        weight_slice,
                        path_out_dir / f'{file_stem}_overlay_slice.png',
                        normalize_slice=True,
                    )
                    # Plane × slice product (legacy name _overlay.png)
                    _save_attention_overlay(
                        source, weight, path_out_dir / f'{file_stem}_overlay.png',
                    )
                    if dataset in ['LIDC']:
                        save_image(
                            tensor_cam2image(
                                minmax_norm(source),
                                minmax_norm(batch['mask'].detach().cpu()),
                                alpha=0.5,
                            ),
                            str(path_out_dir / f"{file_stem}_overlay_gt.png"),
                            normalize=False,
                        )
                except Exception as exc:
                    logger.error("Failed saving attention maps for UID %s: %s", uid, exc)
                else:
                    if n == 0 or (n + 1) % 25 == 0 or n + 1 == n_cases:
                        logger.info("Attention progress: %d/%d (UID %s)", n + 1, n_cases, uid)
                        progress.set_postfix(uid=uid, refresh=False)
                
        else:
            # Run prediction 
            # path_out_dir = path_out/'embeddings-correct-fulldata'
            # path_out_dir.mkdir(exist_ok=True, parents=True)
            pred, _, _, _ = run_pred(model, batch, save_attn=False, use_softmax=use_tta, use_tta=use_tta)
            # pred = pred.detach().cpu()
            # np.savez(path_out_dir/f"{uid}_pred.npz", pred=pred)
            

        pred_prob = torch.softmax(pred.cpu(), dim=-1)
        pred_binary = torch.argmax(pred_prob, dim=1)
        pred_scores = pred_prob[:, 1]
        logger.debug(f"Inside loop. UID: {uid}, target shape: {target.shape}, current len(results): {len(results)}")
        if target.shape[0] == 0:
            logger.warning(f"UID {uid}: target.shape[0] is 0 for this item/batch. No entries will be added to 'results'.")
       

        results.extend([{
            'UID': uid,
            'GT': target[b].item(),
            'NN': pred_binary[b].item(),
            'NN_pred': pred_scores[b].item()
        } for b in range(target.shape[0])] )

    if get_segmentation:
        df_seg = pd.DataFrame(results_seg)
        df = pd.DataFrame(results)
        df = pd.merge(df_seg, df, on="UID")
        print("Save segmentation results to ", path_out)
        df.to_csv(path_out/'results_seg.csv', index=False)
        logger.info(f"Dice: {df['Dice'].mean():.2f}±{df['Dice'].std():.2f}") 
        logger.info(f"IOU: {df['IOU'].mean():.2f}±{df['IOU'].std():.2f}") 
        logger.info(f"ASSD: {df['ASSD'].mean():.2f}±{df['ASSD'].std():.2f}") 
        logger.info(f"Dice: {df['Dice_foreground'].mean():.2f}±{df['Dice_foreground'].std():.2f}") 
        logger.info(f"IOU: {df['IOU_foreground'].mean():.2f}±{df['IOU_foreground'].std():.2f}") 
        logger.info(f"ASSD: {df['ASSD_foreground'].mean():.2f}±{df['ASSD_foreground'].std():.2f}")

    elif save_embeddings:
        idx_path = path_out / 'embeddings_fused' / 'index.csv'
        pd.DataFrame(embedding_rows).to_csv(idx_path, index=False)
        logger.info("Saved %d embedding files under %s (see %s).", len(embedding_rows), path_out / 'embeddings_fused', idx_path)
        logger.info(
            "Each npz stores key 'pred' float32 shaped [batch, embed_dim]; "
            "for DinoClassifierSlice this matches DinoClassifierSlice.forward(..., without_linear=True), "
            "i.e. after slice fusion CLS, before Linear."
        )

    elif get_attention:
        pd.DataFrame(classification).to_csv(path_out / 'classification.csv', index=False)
        attn_dir = path_out / 'attention-rotated'
        n_overlays = len(list(attn_dir.glob('*_overlay.png'))) if attn_dir.is_dir() else 0
        logger.info(
            "Attention complete: %d cases logged, %d overlay PNGs in %s",
            len(classification),
            n_overlays,
            attn_dir,
        )

    elif not get_attention:
        df = pd.DataFrame(results)
        df.to_csv(path_out / results_csv_name, index=False)


        acc = accuracy_score(df['GT'], df['NN'])
        logger.info(f"Acc: {acc:.2f}") 

        #  -------------------------- Confusion Matrix -------------------------
        cm = confusion_matrix(df['GT'], df['NN'], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n = len(df)
        logger.info("Confusion Matrix: TN {} ({:.2f}%), FP {} ({:.2f}%), FN {} ({:.2f}%), TP {} ({:.2f}%)".format(tn, tn/n*100, fp, fp/n*100, fn, fn/n*100, tp, tp/n*100 ))


        # ------------------------------- ROC-AUC ---------------------------------
        fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6)) 
        tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(df['GT'], df['NN_pred'], axis, fontdict=fontdict)
        fig.tight_layout()
        fig.savefig(path_out / roc_png_name, dpi=300)
        logger.info("AUC {:.2f}".format(auc_val))


        #  -------------------------- Confusion Matrix -------------------------
        acc = cm2acc(cm)
        _,_, sens, spec = cm2x(cm)
        df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
        fig, axis = plt.subplots(1, 1, figsize=(4,4))
        sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True) 
        axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
        axis.set_xlabel('Prediction' , fontdict=fontdict)
        axis.set_ylabel('True' , fontdict=fontdict)
        fig.tight_layout()
        fig.savefig(path_out / cm_png_name, dpi=300)

        logger.info(f"Malign  Objects: {np.sum(df['GT'])}")
        logger.info("Confusion Matrix {}".format(cm))
        logger.info("Sensitivity {:.2f}".format(sens))
        logger.info("Specificity {:.2f}".format(spec))

