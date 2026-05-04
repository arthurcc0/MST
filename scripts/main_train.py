import sys
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# # Add DinoV3 path to sys.path so it can be imported as 'dinov3'
# dinov3_path = os.path.join(project_root, 'mst', 'models', 'extern')
# if dinov3_path not in sys.path:
#     sys.path.insert(0, dinov3_path)
import argparse
from pathlib import Path
from datetime import datetime
import wandb 
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_penn import PENN_DataModule, PENN_Dataset3D
from mst.data.datamodules import DataModule
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoClassifierSlice, DinoClassifierPaired, DinoTxtClassifier



def get_model(name, **kwargs):
    if name == 'ResNet':
        return ResNet(in_ch=1, out_ch=2, spatial_dims=3, **kwargs)
    elif name == 'ResNetSliceTrans':
        return ResNetSliceTrans(in_ch=1, out_ch=2, spatial_dims=2, **kwargs)
    elif name == 'DinoClassifierSlice':
        return DinoClassifierSlice(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
    elif name == 'DinoClassifierPaired':
        return DinoClassifierPaired(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
    elif name == 'DinoTxtClassifier':
        return DinoTxtClassifier(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

from huggingface_hub import login
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    # Hugging Face Login
    token = os.getenv("HUGGING_FACE_TOKEN")
    if token:
        login(token=token)
        print("Successfully logged in to Hugging Face.")
    else:
        print("Hugging Face token not found. Skipping login.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DUKE', choices=['DUKE', 'LIDC', 'MRNet', 'PENN'])
    parser.add_argument('--model_name', type=str, default='DinoClassifierSlice', choices=['ResNet', 'ResNetSliceTrans', 'DinoClassifierSlice', 'DinoClassifierPaired', 'DinoTxtClassifier'])
    parser.add_argument('--model_version', type=str, default='v2', choices=['v2', 'v3'], help='DINO model version.')
    parser.add_argument('--path_root_output', type=str, default='./runs', help="Root output path")
    parser.add_argument('--only_malignants', action='store_true', default=False, help="Use only malignant samples for PENN dataset")
    parser.add_argument('--with_laterality', action='store_true', default=False, help="Use only samples with laterality for PENN dataset") 
    parser.add_argument('--paired_sampling', type=lambda x: (str(x).lower() == 'true'), nargs='?', const=True, default=False, help='Enable paired sampling.') 
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloaders.')
    parser.add_argument('--clinical_notes_path', type=str, default='./malignancy_reports/malignancy_summary_table.csv', help='Path to clinical notes CSV file.')
    parser.add_argument('--use_clinical_notes', action='store_true', default=False, help='Enable clinical notes for multimodal training.')
    parser.add_argument('--input_type', type=str, default='subtraction',
                        help="Tag describing the input images (e.g. 'subtraction', 'pre', 'post'). Used in the run directory name and wandb run name.")
    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    input_tag = str(args.input_type).strip().replace(' ', '_') or 'subtraction'
    path_run_dir = Path(args.path_root_output) / args.dataset / f'{args.model_name}_{current_time}_{input_tag}_multi'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu'
    torch.set_float32_matmul_precision('high')

    # ------------ Initialize Data ----------------
    if args.dataset == 'PENN':
        dm = PENN_DataModule(**vars(args))
    elif args.dataset == 'DUKE':
        # Filter arguments for DUKE_Dataset3D constructor
        duke_kwargs = {
            'path_root': getattr(args, 'path_root', None),
            'fold': getattr(args, 'fold', 0),
            'fraction': getattr(args, 'fraction', None),
            'transform': None,
            'image_resize': getattr(args, 'image_resize', None),
            'resample': getattr(args, 'resample', None),
            'flip': getattr(args, 'flip', False),
            'random_rotate': getattr(args, 'random_rotate', False),
            'image_crop': getattr(args, 'image_crop', (224, 224, 32)),
            'random_center': getattr(args, 'random_center', False),
            'noise': getattr(args, 'noise', False),
            'to_tensor': True,
            'get_segmentation': getattr(args, 'get_segmentation', False),
            'clinical_notes_path': args.clinical_notes_path,
            'use_clinical_notes': args.use_clinical_notes or args.model_name == 'DinoTxtClassifier'
        }
        
        # Create train, val, and test datasets
        ds_train = DUKE_Dataset3D(split='train', **duke_kwargs)
        ds_val = DUKE_Dataset3D(split='val', **duke_kwargs)
        ds_test = DUKE_Dataset3D(split='test', **duke_kwargs)
        
        # Create DataModule with the datasets
        dm = DataModule(
            ds_train=ds_train,
            ds_val=ds_val,
            ds_test=ds_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        # This part can be extended to support other datasets with their own DataModules
        raise NotImplementedError(f"DataModule for {args.dataset} is not implemented yet.")

    accumulate_grad_batches = 2

# ------------ Initialize Model ------------
    model_name = 'DinoClassifierPaired' if args.paired_sampling else args.model_name
    model = get_model(model_name, **vars(args))
    
    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 50
    logger = WandbLogger(
        project=f'Classifier_{args.dataset}_MST',
        name=f'{type(model).__name__}_{input_tag}',
        log_model=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=10,
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=1,
        mode=min_max,
        save_weights_only=True,
    )
    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, lr_monitor, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=200,
        max_epochs=1000,
        num_sanity_val_steps=2,
        logger=logger
    )

    # ---------------- Load weights if ckpt_path is provided ----------------
    if args.ckpt_path:
        print(f"Loading weights from checkpoint: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        # Use the custom loader for the paired model, and standard loading for others
        if isinstance(model, DinoClassifierPaired):
            model.load_weights(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    # ---------------- Execute Training ----------------
    # Pass ckpt_path=None because we have already loaded the weights
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)

    wandb.finish(quiet=True)