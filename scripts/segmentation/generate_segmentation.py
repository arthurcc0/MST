import sys
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
# Add the project root to sys.path if it's not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import logging
from tqdm import tqdm
import math 
import torch 
import torchio as tio
import numpy as np 
import torch.nn.functional as F

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoClassifierSlice
from mst.models.utils.functions import one_hot

def get_dataset(name, split, **kwargs):
    if name == 'DUKE':
        return DUKE_Dataset3D(split=split, **kwargs)
    elif name == 'LIDC':
        return LIDC_Dataset3D(split=split, **kwargs)
    elif name == 'MRNet':
        return MRNet_Dataset3D(split=split, **kwargs)
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

def _pred_trans(model, source, src_key_padding_mask, save_attn=False, use_softmax=True):
    # Run model
    if isinstance(model, ResNetSliceTrans) and save_attn:
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn)
    else:
        with torch.no_grad():
            pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn)

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

def run_pred(model, batch, save_attn=False, use_softmax=True, use_tta=False):
    source, src_key_padding_mask = batch['source'], batch.get('src_key_padding_mask', None)
    pred_func = None 
    if isinstance(model, ResNetSliceTrans): 
        pred_func = _pred_trans
    elif isinstance(model, ResNet):
        pred_func = _pred_resnet
    elif isinstance(model, DinoClassifierSlice):
        pred_func = _pred_trans

    pred, weight, weight_slice = pred_func(model, source, src_key_padding_mask, save_attn=save_attn, use_softmax=use_softmax)    

    if use_tta:
        for flip_dim in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4),]:
            pred_i, weight_i, weight_slice_i = pred_func(model, torch.flip(source, flip_dim), src_key_padding_mask, save_attn=save_attn, use_softmax=use_softmax)
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
    parser = argparse.ArgumentParser(description="Generate segmentations from a trained model.")
    parser.add_argument('--run_folder', required=True, type=str, help='Path to the run folder (e.g., LIDC/ResNet_1234).')
    parser.add_argument('--output_dir', default=None, type=str, help='Directory to save the segmentation files. Defaults to input_dir if not provided.')
    parser.add_argument('--use_tta', action='store_true', help='Use test time augmentation.')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory containing NIfTI files to segment. Overrides dataset loading.')
    parser.add_argument('--output_seg_filename', type=str, default='seg.nii.gz', help='Filename for the output segmentation.')

    args = parser.parse_args()
    use_tta = args.use_tta
    print(f"Using TTA: {use_tta}")

    run_folder = Path(args.run_folder)
    dataset_name = run_folder.parent.name
    model_name = run_folder.name.split('_', 1)[0]
    
    # Special handling for DinoV2 model name
    if 'ResNetSliceTrans_2025_06_06_192329_multi' in str(run_folder):
        print(f"Overriding model_name from '{model_name}' to 'DinoV2ClassifierSlice' for run: {run_folder}")
        model_name = "DinoV2ClassifierSlice"

    #------------ Settings/Defaults ----------------
    path_run = run_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    # ------------ Initialize Model ------------
    model = get_model(model_name).load_best_checkpoint(path_run)
    model.to(device)
    model.eval()

    # ------------ Determine execution path (dataset vs. directory) ------------
    if args.input_dir:
        # --- Process a directory of NIfTI files ---
        input_path = Path(args.input_dir)
        output_path = Path(args.output_dir) if args.output_dir else input_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(output_path / f'{Path(__file__).stem}.log', mode='w')])
        logger = logging.getLogger(__name__)

        nifti_files = sorted(list(input_path.glob('*.nii.gz')))
        if not nifti_files:
            logger.warning(f"No .nii.gz files found in {input_path}")
        
        logger.info(f"Starting segmentation for {len(nifti_files)} files in {input_path}...")

        for n, file_path in enumerate(tqdm(nifti_files, desc="Generating Segmentations")):
            uid = file_path.stem
            try:
                image = tio.ScalarImage(file_path)
                batch = {
                    'source': image.data.unsqueeze(0).to(device),
                    'affine': [image.affine],
                    'uid': [uid],
                    'src_key_padding_mask': None
                }
                
                _, weight, _ = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)
                if weight is None:
                    logger.warning(f"Could not generate attention weight for {uid}. Skipping.")
                    continue

                weight = weight.detach().cpu()
                seg_tensor = (weight > np.quantile(weight, 0.90)).type(torch.int16)
                
                affine = batch['affine'][0]
                seg_image = tio.ScalarImage(tensor=seg_tensor[0], affine=affine)
                
                output_filename = output_path / args.output_seg_filename
                seg_image.save(output_filename)
                logger.info(f"Saved segmentation for {uid} to {output_filename}")

            except Exception as e:
                logger.error(f"Failed to process {uid}. Error: {e}", exc_info=True)

    else:
        # --- Process a dataset ---
        path_out = Path(args.output_dir or './segmentations') / run_folder
        path_out.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(path_out / f'{Path(__file__).stem}.log', mode='w')])
        logger = logging.getLogger(__name__)

        ds_test = get_dataset(name=dataset_name, split='test', get_segmentation=False)
        dm = DataModule(ds_test=ds_test, batch_size=1, num_workers=8, pin_memory=True)
        
        logger.info(f"Starting segmentation generation for {len(dm.test_dataloader())} cases...")

        for n, batch in enumerate(tqdm(dm.test_dataloader(), desc="Generating Segmentations")):
            uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())
            try:
                _, weight, _ = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)
                if weight is None:
                    logger.warning(f"Could not generate attention weight for UID: {uid}. Skipping.")
                    continue

                weight = weight.detach().cpu()
                seg_tensor = (weight > np.quantile(weight, 0.999)).type(torch.int16)
                
                affine = batch['affine'][0].numpy()
                seg_image = tio.ScalarImage(tensor=seg_tensor[0], affine=affine)
                
                output_filename = path_out / f"{uid}_segmentation.nii.gz"
                seg_image.save(output_filename)
                logger.info(f"Saved segmentation for UID: {uid} to {output_filename}")

            except Exception as e:
                logger.error(f"Failed to process UID: {uid}. Error: {e}", exc_info=True)

    logger.info("Segmentation generation complete.")