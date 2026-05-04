import sys
import os
import random
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
# Get the directory containing the script (e.g., /path/to/MST/scripts)
script_dir = os.path.dirname(script_path)
# Get the project root directory (e.g., /path/to/MST)
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pathlib import Path
import argparse
import logging
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
from mst.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from mst.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot
def set_seed(seed=42):
    random.seed(seed)                        # Python random module
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed(seed)             # PyTorch GPU
    torch.cuda.manual_seed_all(seed)         # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()


def get_dataset(name, split, run_folder, **kwargs):
    import yaml
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
        path_csv = Path(penn_split_csv) if penn_split_csv else PENN_Dataset3D.default_split_csv_path()

        config_file = Path(run_folder) / 'config.yaml'
        fold = None
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    config = {}
            fold = config.get('fold', 0)
            cfg_csv = config.get('penn_split_csv') or config.get('split_csv_path')
            if penn_split_csv is None and cfg_csv:
                path_csv = Path(cfg_csv)
        if fold is None:
            print(f"Warning: config.yaml not found in {run_folder}. Defaulting to fold=0.")
            fold = 0

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
        return pred, None, None 

    # Spatial attention     
    weight = model.get_attention_maps()  # [B*D, Heads, HW]
    weight = weight.mean(dim=1) # Mean of heads 
    spatial_shape = weight.shape[-2:] if isinstance(model, ResNetSliceTrans) else torch.tensor(source.shape[3:])//14 
    weight = weight.view(1, 1, source.shape[2], *spatial_shape)

    # Slice attention 
    weight_slice = model.get_slice_attention() # [B*D, Heads, 1]
    weight_slice = weight_slice.mean(dim=1) # Mean of heads 
    weight_slice = weight_slice.view(1, 1, -1, 1, 1)*torch.ones_like(source, device=weight.device)
    return pred, weight, weight_slice




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
        return pred, None, None 

    weight = model.get_attention_maps()

    # Slice attention (dummy)
    weight_slice = torch.ones_like(source, device=weight.device)

    return pred, weight, weight_slice


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
    elif isinstance(model, DinoClassifierSlice):
        pred_func = _pred_trans

    model_forward_kwargs = {}
    if embeddings_only and isinstance(model, DinoClassifierSlice):
        model_forward_kwargs['without_linear'] = True
        use_softmax = False

    pred, weight, weight_slice = pred_func(
        model,
        source,
        src_key_padding_mask,
        save_attn=save_attn,
        use_softmax=use_softmax,
        model_forward_kwargs=model_forward_kwargs,
    )

    if use_tta:
        for flip_dim in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4),]:
            pred_i, weight_i, weight_slice_i = pred_func(
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

        pred = pred / 8
        if save_attn:
            weight = weight / 8
            weight_slice = weight_slice / 8

    # Interpolate to required size 
    if save_attn:
        weight = F.interpolate(weight, size=source.shape[2:], mode='trilinear')

    return pred, weight, weight_slice 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default='./runs', type=str)
    parser.add_argument('--run_folder', default='LIDC/ResNet', type=str)
    parser.add_argument(
        '--penn_split_csv',
        default=None,
        type=str,
        help=(
            'PENN splits CSV (columns include Fold, Split, UID, …). '
            'Default: PENN_Dataset3D.default_split_csv_path(); '
            'overridden by config.yaml keys penn_split_csv or split_csv_path when present.'
        ),
    )
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--get_attention', action='store_true', help='Flag to get attention')
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
    get_segmentation = args.get_segmentation
    use_tta = args.use_tta
    save_embeddings = args.save_embeddings
    print(f"Using TTA {use_tta}")
    print(f"Save embeddings {save_embeddings}")

    run_folder = Path(args.run_folder)
    dataset = run_folder.parent.name
    logger = logging.getLogger(__name__)
    model_name = run_folder.name.split('_', 1)[0]
    if 'DinoV2ClassifierSlice_2025_08_25_190331_multi' in str(run_folder):
        logger.info(
            f"Overriding model_name from '{model_name}' to 'DinoClassifierSlice' "
            f"(legacy run folder) for run: {run_folder}"
        )
        model_name = "DinoClassifierSlice"
    #------------ Settings/Defaults ----------------
    path_run = run_folder
    results_folder = 'results_tta'if use_tta else 'results-duke-penn-training'
    path_out = Path(args.output_dir)/results_folder/run_folder
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}
    torch.set_float32_matmul_precision('high')

    # ------------ Logging --------------------
    logger.setLevel(logging.INFO) 
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(path_out / f'{Path(__file__).name}.txt', mode='w'))

    # ------------ Load Data ----------------
    ds_test = get_dataset(
        name=dataset,
        split='test',
        run_folder=args.run_folder,
        get_segmentation=get_segmentation,
        penn_split_csv=args.penn_split_csv,
    )

    dm = DataModule(
        ds_test=ds_test,
        batch_size=1, 
        num_workers=0,
        pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = get_model(model_name).load_best_checkpoint(path_run)
    # Legacy path extracts penultimate embeddings by replacing the head with Identity.
    if not save_embeddings:
        model.linear = nn.Identity()

    model.to(device)
    model.eval()

    classification = []
    results = []
    results_seg = []
    counter = 0 
    embedding_rows = []
    for n, batch in enumerate(tqdm(dm.test_dataloader())):
        logger.debug(f"Processing batch {n}, Keys: {list(batch.keys())}, Target: {batch.get('target', 'N/A')}, UID: {batch.get('uid', 'N/A')}")
        batch['source'] = batch['source'].to(device).float()
        source, target = batch['source'], batch['target']
        uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())

        if save_embeddings:
            path_emb = path_out / 'embeddings_fused'
            path_emb.mkdir(parents=True, exist_ok=True)
            pred_emb, _, _ = run_pred(
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
            pred, weight, weight_slice = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)
            
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
            # Output folder  
            path_out_dir = path_out/'attention-rotated'
            path_out_dir.mkdir(parents=True, exist_ok=True)
        
            # Skip cases without target label 
            # Only eval limited number
            #counter += 1
            #if counter > 5:
            #    break 

            # Run prediction 
            pred, weight, weight_slice = run_pred(model, batch, save_attn=True, use_softmax=False, use_tta=use_tta)

           
            # Clip  
            weight_slice = weight_slice.detach().cpu()
            weight_slice /= weight_slice.sum()

            weight = weight.detach().cpu()
            weight = weight.clip(*np.quantile(weight, [0.995, 0.999]))
            pred = pred.detach().cpu()

            pred_binary = torch.argmax(pred, dim=1)
            pred_prob = torch.softmax(pred, dim=-1)[:, 1]
            result = {
                'UID': uid,
                'GT': target.item(),
                'NN': pred_binary.item(),
                'NN_pred': pred_prob.item()
            }
            classification.append(result)
            pd.DataFrame(classification).to_csv(path_out/'classification.csv', index=False)

            
            # Save 
            save_image(tensor2image(source.rot90(2, (2, 3))), path_out_dir/f'{uid}_input.png', normalize=True)
            source_for_viz = source.mean(dim=1, keepdim=True)
            save_image(tensor_cam2image(minmax_norm(source_for_viz.rot90(2, (2, 3))), minmax_norm(weight.rot90(2, (2, 3))), alpha=0.5), 
                        path_out_dir/f"{uid}_overlay.png", normalize=False)
            save_image(tensor_cam2image(minmax_norm(source_for_viz.rot90(2, (2, 3))), minmax_norm(weight_slice.rot90(2, (2, 3))), alpha=0.5), 
                        path_out_dir/f"{uid}_overlay_slice.png", normalize=False)
            if dataset in ['LIDC']:
                save_image(tensor_cam2image(minmax_norm(source), minmax_norm(batch['mask'].detach().cpu()), alpha=0.5),
                            path_out_dir/f"{uid}_overlay_gt.png", normalize=False) 
                
        else:
            # Run prediction 
            # path_out_dir = path_out/'embeddings-correct-fulldata'
            # path_out_dir.mkdir(exist_ok=True, parents=True)
            pred, _, _ = run_pred(model, batch, save_attn=False, use_softmax=use_tta, use_tta=use_tta)
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

    elif not get_attention:
        df = pd.DataFrame(results)
        df.to_csv(path_out/'results.csv', index=False)


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
        fig.savefig(path_out/f'roc.png', dpi=300)
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
        fig.savefig(path_out/f'confusion_matrix.png', dpi=300)

        logger.info(f"Malign  Objects: {np.sum(df['GT'])}")
        logger.info("Confusion Matrix {}".format(cm))
        logger.info("Sensitivity {:.2f}".format(sens))
        logger.info("Specificity {:.2f}".format(spec))

