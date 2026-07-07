import sys
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
scripts_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# # Add DinoV3 path to sys.path so it can be imported as 'dinov3'
# dinov3_path = os.path.join(project_root, 'mst', 'models', 'extern')
# if dinov3_path not in sys.path:
#     sys.path.insert(0, dinov3_path)
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import wandb 
import torch 
from sklearn.utils.class_weight import compute_class_weight
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
from mst.models.dino_v3 import DinoClassifierSliceV3
from mst.losses.focal_loss import FocalLoss
from mst.utils.random_seed import set_random_seed


def _train_malignant_labels(dataset, label_col: str = 'Malignant') -> np.ndarray:
    """Integer class labels from a dataset's underlying train split table."""
    if not hasattr(dataset, 'df'):
        raise TypeError(f"Cannot read labels from {type(dataset).__name__} (no .df attribute).")
    return dataset.df[label_col].astype(int).values


def _balanced_cross_entropy_weights(
    labels: np.ndarray,
    n_classes: int = 2,
) -> torch.Tensor:
    """sklearn 'balanced' weights indexed by class id (0 .. n_classes-1)."""
    labels = np.asarray(labels, dtype=int)
    present = np.unique(labels)
    if len(present) < 2:
        print(
            f"Warning: training set has a single class {present.tolist()}; "
            "using uniform CrossEntropyLoss weights."
        )
        return torch.ones(n_classes, dtype=torch.float32)
    raw = compute_class_weight(class_weight='balanced', classes=present, y=labels)
    weight = torch.ones(n_classes, dtype=torch.float32)
    for cls, w in zip(present, raw):
        weight[int(cls)] = float(w)
    return weight


def _prevalence_class_weights(labels: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    """Per-class weight proportional to class frequency (for focal alpha)."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=n_classes)
    total = max(int(counts.sum()), 1)
    return torch.tensor([counts[i] / total for i in range(n_classes)], dtype=torch.float32)


def _build_loss(
    args,
    dm: PENN_DataModule | None,
    *,
    paired_model: bool,
) -> tuple[type, dict]:
    """Return (loss_class, loss_kwargs) for BasicClassifier."""
    loss_name = args.loss.strip().lower()
    if loss_name == "ce":
        if args.class_weight.strip().lower() == "balanced" and not paired_model:
            if dm is None:
                raise ValueError("PENN DataModule required for balanced CE weights.")
            dm.setup("fit")
            labels = _train_malignant_labels(dm.train_dataset)
            weight = _balanced_cross_entropy_weights(labels, n_classes=2)
            print(
                f"CrossEntropyLoss class weights (balanced, train fold {args.fold}): "
                f"benign(0)={weight[0]:.4f}, malignant(1)={weight[1]:.4f}"
            )
            return torch.nn.CrossEntropyLoss, {"weight": weight}
        return torch.nn.CrossEntropyLoss, {}

    if loss_name != "focal":
        raise ValueError(f"Unknown --loss {args.loss!r}; use 'ce' or 'focal'.")

    if paired_model:
        raise ValueError("Focal loss is not supported for DinoClassifierPaired (BCE).")

    alpha_mode = args.focal_alpha.strip().lower()
    alpha = None
    if alpha_mode != "none":
        if dm is None:
            raise ValueError("PENN DataModule required for focal alpha from training labels.")
        dm.setup("fit")
        labels = _train_malignant_labels(dm.train_dataset)
        if alpha_mode == "balanced":
            alpha = _balanced_cross_entropy_weights(labels, n_classes=2)
            print(
                f"FocalLoss alpha (balanced, train fold {args.fold}): "
                f"benign(0)={alpha[0]:.4f}, malignant(1)={alpha[1]:.4f}"
            )
        elif alpha_mode == "prevalence":
            alpha = _prevalence_class_weights(labels, n_classes=2)
            print(
                f"FocalLoss alpha (prevalence, train fold {args.fold}): "
                f"benign(0)={alpha[0]:.4f}, malignant(1)={alpha[1]:.4f}"
            )
        else:
            raise ValueError(
                f"Unknown --focal-alpha {args.focal_alpha!r}; use none, balanced, or prevalence."
            )

    if args.class_weight.strip().lower() == "balanced":
        print("Note: --class_weight balanced is ignored when --loss focal (use --focal-alpha balanced).")

    loss_kwargs: dict = {"gamma": float(args.focal_gamma), "reduction": "mean"}
    if alpha is not None:
        loss_kwargs["alpha"] = alpha.tolist()
    print(f"Using FocalLoss(gamma={args.focal_gamma}, alpha={alpha_mode})")
    return FocalLoss, loss_kwargs


def _resolve_train_precision(precision_arg: str, model_name: str) -> str:
    resolved = precision_arg
    if resolved == 'auto':
        resolved = 'bf16-mixed' if model_name == 'DinoClassifierSliceV3' else '16-mixed'
    if model_name == 'DinoClassifierSliceV3' and resolved == '16-mixed':
        print(
            "Warning: 16-mixed produces NaN logits with DINOv3; using bf16-mixed instead."
        )
        resolved = 'bf16-mixed'
    return resolved


def get_model(name, **kwargs):
    if name == 'ResNet':
        return ResNet(in_ch=1, out_ch=2, spatial_dims=3, **kwargs)
    elif name == 'ResNetSliceTrans':
        return ResNetSliceTrans(in_ch=1, out_ch=2, spatial_dims=2, **kwargs)
    elif name == 'DinoClassifierSlice':
        return DinoClassifierSlice(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
    elif name == 'DinoClassifierSliceV3':
        return DinoClassifierSliceV3(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
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
    parser.add_argument('--model_name', type=str, default='DinoClassifierSlice', choices=['ResNet', 'ResNetSliceTrans', 'DinoClassifierSlice', 'DinoClassifierSliceV3', 'DinoClassifierPaired', 'DinoTxtClassifier'])
    parser.add_argument('--model_version', type=str, default='v2', choices=['v2', 'v3'], help='DINO model version (v3 only for DinoTxtClassifier; slice models use --model_name DinoClassifierSliceV3).')
    parser.add_argument(
        '--model_size',
        type=str,
        default='s',
        choices=['s', 'b', 'l', 'g'],
        help='DINO ViT backbone size: s/b/l for DINOv2 and DINOv3; g is DINOv2 giant only (default: s).',
    )
    parser.add_argument('--use_registers', type=lambda x: str(x).lower() == 'true', nargs='?', const=True, default=False,help='Load DINOv2 backbone with 4 register tokens (Darcet et al., 2023).')
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
    parser.add_argument('--slices', type=int, default=32,
                        help='Number of slices (depth D) used for image_crop=(224, 224, slices). Must be <= the depth produced by offline preprocessing.')
    parser.add_argument('--path_root_data', type=str, default=None,
                        help="Override for the preprocessed-data folder (PENN). Accepts a bare folder name "
                             "resolved under PENN_Dataset3D.AUX_PATH (e.g. 'final_cropped_and_masked_data_d64') "
                             "or an absolute path. Defaults to 'final_cropped_and_masked_data'.")
    parser.add_argument(
        '--penn_split_csv',
        type=str,
        default=None,
        help=(
            "PENN train/val split CSV (Fold, Split, UID, Malignant, …). "
            "Filename under PENN_Dataset3D.AUX_PATH or absolute path. "
            "Default: new_penn_datasplit.csv. Use old_penn_datasplit.csv for legacy cohort."
        ),
    )
    parser.add_argument(
        '--cohort',
        type=str,
        default=None,
        help=(
            "Short tag for this training stage (e.g. old_penn, new_penn). "
            "Included in the run folder and wandb run name so sequential fine-tunes are easy to tell apart."
        ),
    )
    parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold index in the PENN split CSV.')
    parser.add_argument(
        '--freeze_backbone',
        action='store_true',
        help='Freeze DINOv2 weights; train only slice-fusion MST + classifier (recommended for small new-Penn stage).',
    )
    parser.add_argument(
        '--unfreeze_encoder_blocks',
        type=int,
        default=0,
        help='After loading --ckpt_path, unfreeze the last N ViT blocks (DINOv2 or DINOv3; 0 = keep backbone fully frozen if --freeze_backbone).',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='LR for trainable head/MST modules (slice fusion + linear). Default: model default (1e-6). Try 1e-4 with --freeze_backbone on new Penn.',
    )
    parser.add_argument(
        '--encoder_lr',
        type=float,
        default=None,
        help='LR for unfrozen ViT blocks when --unfreeze_encoder_blocks > 0. Default: 0.1 * learning_rate.',
    )
    parser.add_argument(
        '--class_weight',
        type=str,
        default='none',
        choices=['none', 'balanced'],
        help=(
            "CrossEntropyLoss class weights from the training split. "
            "'balanced' uses sklearn balanced weights (minority up-weighted). "
            "Ignored for DinoClassifierPaired (BCE). Ignored when --loss focal. "
            "Default: none."
        ),
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='ce',
        choices=['ce', 'focal'],
        help="Training loss: cross-entropy (default) or focal loss.",
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help="Focal loss gamma when --loss focal (default: 2.0).",
    )
    parser.add_argument(
        '--focal-alpha',
        type=str,
        default='balanced',
        choices=['none', 'balanced', 'prevalence'],
        help=(
            "Focal loss per-class alpha from train-fold labels when --loss focal. "
            "'balanced' matches sklearn balanced weights; 'prevalence' uses class fractions. "
            "Default: balanced."
        ),
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='auto',
        choices=['auto', '16-mixed', 'bf16-mixed', '32-true'],
        help=(
            "Lightning precision mode. 'auto' uses bf16-mixed for DinoClassifierSliceV3 "
            "(fp16-mixed NaNs on the HF DINOv3 backbone) and 16-mixed otherwise."
        ),
    )
    parser.add_argument(
        '--slab_tissue_soft_weight',
        action='store_true',
        help=(
            'PENN: scale slab embeddings before slice fusion. Slabs with tissue >= '
            'keep_ratio * max(tissue) keep weight 1.0; thinner slabs ramp down to '
            'min_weight.'
        ),
    )
    parser.add_argument(
        '--slab_tissue_min_weight',
        type=float,
        default=0.1,
        help='PENN: floor for --slab_tissue_soft_weight (default: 0.1).',
    )
    parser.add_argument(
        '--slab_tissue_keep_ratio',
        type=float,
        default=0.5,
        help=(
            'PENN: slabs with tissue fraction >= this fraction of the richest slab '
            'keep weight 1.0 (default: 0.5). Only thinner slabs are down-weighted.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for Python/NumPy/PyTorch and DataLoader shuffling (default: 42).',
    )
    parser.add_argument(
        '--deterministic',
        type=str,
        default='false',
        choices=['true', 'warn', 'false'],
        help=(
            "PyTorch Lightning deterministic mode (default: false). "
            "Use --seed for reproducible init + data order. "
            "'warn'/'true' enable stricter CUDA ops and may warn or fail on val AUC."
        ),
    )
    args = parser.parse_args()

    deterministic_mode = {'true': True, 'warn': 'warn', 'false': False}[args.deterministic]

    set_random_seed(args.seed, cudnn_deterministic=(deterministic_mode is True))
    print(
        f"Random seed: {args.seed} "
        f"(cuDNN deterministic={deterministic_mode is True}, "
        f"Trainer deterministic={deterministic_mode!r})"
    )

    # Inject image_crop derived from --slices so it flows through to the dataset
    # (PENN_DataModule and the explicit DUKE branch both pick up image_crop from args).
    args.image_crop = (224, 224, int(args.slices))

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    input_tag = str(args.input_type).strip().replace(' ', '_') or 'subtraction'
    registers_tag = '_reg' if args.use_registers else ''
    v3_tag = '_v3' if args.model_name == 'DinoClassifierSliceV3' else ''
    # Encode the slice count in the run directory name so prediction can auto-detect it.
    slices_tag = '' if args.slices == 32 else f'_d{args.slices}'
    multi_tag = '_multi' if args.use_clinical_notes else ''
    cohort_tag = ''
    if args.cohort:
        cohort_tag = '_' + str(args.cohort).strip().replace(' ', '_')
    path_run_dir = Path(args.path_root_output) / args.dataset / (
        f'{args.model_name}_{current_time}_{input_tag}{cohort_tag}{v3_tag}{registers_tag}{slices_tag}{multi_tag}'
    )
    path_run_dir.mkdir(parents=True, exist_ok=True)

    penn_split_csv_path = None
    if args.dataset == 'PENN':
        penn_split_csv_path = PENN_Dataset3D.resolve_split_csv_path(args.penn_split_csv)
        print(f"PENN split CSV: {penn_split_csv_path}")
    accelerator = 'gpu'
    torch.set_float32_matmul_precision('high')

    # ------------ Initialize Data ----------------
    if args.dataset == 'PENN':
        dm = PENN_DataModule(**vars(args))
        train_config = {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'model_size': args.model_size,
            'fold': args.fold,
            'cohort': args.cohort,
            'penn_split_csv': str(penn_split_csv_path),
            'split_csv_path': str(penn_split_csv_path),
            'path_root_data': args.path_root_data,
            'slices': args.slices,
            'input_type': args.input_type,
            'only_malignants': args.only_malignants,
            'ckpt_init': args.ckpt_path,
            'freeze_backbone': args.freeze_backbone,
            'unfreeze_encoder_blocks': args.unfreeze_encoder_blocks,
            'learning_rate': args.learning_rate,
            'encoder_lr': args.encoder_lr,
            'class_weight': args.class_weight,
            'loss': args.loss,
            'focal_gamma': args.focal_gamma,
            'focal_alpha': args.focal_alpha,
            'seed': args.seed,
            'deterministic': args.deterministic,
            'precision': _resolve_train_precision(args.precision, args.model_name),
        }
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
            num_workers=args.num_workers,
            seed=args.seed,
        )
    else:
        # This part can be extended to support other datasets with their own DataModules
        raise NotImplementedError(f"DataModule for {args.dataset} is not implemented yet.")

    accumulate_grad_batches = 1

    paired_model = args.paired_sampling or args.model_name == 'DinoClassifierPaired'
    penn_dm_for_loss = dm if args.dataset == 'PENN' else None
    loss_cls, loss_kwargs = _build_loss(
        args,
        penn_dm_for_loss,
        paired_model=paired_model,
    )
    if args.dataset == 'PENN':
        if loss_kwargs.get("weight") is not None:
            train_config['ce_class_weights'] = [
                float(x) for x in loss_kwargs["weight"].tolist()
            ]
        if loss_kwargs.get("alpha") is not None:
            train_config['focal_alpha_weights'] = [
                float(x) for x in loss_kwargs["alpha"]
            ]
        with open(path_run_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(train_config, f, sort_keys=False)

    # ------------ Initialize Model ------------
    model_name = 'DinoClassifierPaired' if args.paired_sampling else args.model_name
    model_kwargs = vars(args).copy()
    model_kwargs.pop('dataset', None)
    model_kwargs.pop('path_root_output', None)
    model_kwargs.pop('cohort', None)
    model_kwargs.pop('penn_split_csv', None)
    model_kwargs.pop('fold', None)
    model_kwargs.pop('freeze_backbone', None)
    model_kwargs.pop('unfreeze_encoder_blocks', None)
    model_kwargs.pop('learning_rate', None)
    model_kwargs.pop('encoder_lr', None)
    model_kwargs.pop('class_weight', None)
    model_kwargs.pop('loss', None)
    model_kwargs.pop('focal_gamma', None)
    model_kwargs.pop('focal_alpha', None)

    if args.learning_rate is not None:
        model_kwargs['optimizer_kwargs'] = {
            'lr': args.learning_rate,
            'weight_decay': 1e-2,
        }
    if args.encoder_lr is not None:
        model_kwargs['encoder_lr'] = args.encoder_lr
    if model_name in ('DinoClassifierSlice', 'DinoClassifierSliceV3', 'DinoClassifierPaired', 'DinoTxtClassifier'):
        model_kwargs['freeze'] = args.freeze_backbone
    if model_name == 'DinoClassifierSliceV3' and args.use_registers:
        print("Note: --use_registers applies to DINOv2 only; DINOv3 always uses 4 register tokens.")
    model_kwargs['loss'] = loss_cls
    model_kwargs['loss_kwargs'] = loss_kwargs

    model = get_model(model_name, **model_kwargs)
    
    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 50
    logger = WandbLogger(
        project=f'Classifier_{args.dataset}_MST',
        name=f'{type(model).__name__}_{input_tag}{cohort_tag}{registers_tag}{slices_tag}',
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
    train_precision = _resolve_train_precision(args.precision, model_name)
    print(f"Trainer precision: {train_precision}")

    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=train_precision,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, lr_monitor, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=200,
        max_epochs=1000,
        num_sanity_val_steps=2,
        logger=logger,
        deterministic=deterministic_mode,
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

    if args.unfreeze_encoder_blocks > 0 and hasattr(model, 'unfreeze_encoder_last_blocks'):
        n = model.unfreeze_encoder_last_blocks(args.unfreeze_encoder_blocks)
        print(f"Unfroze last {n} encoder block(s); encoder_lr={model.encoder_lr}")

    if args.freeze_backbone or args.unfreeze_encoder_blocks > 0:
        enc_train = sum(
            1 for n, p in model.named_parameters() if n.startswith('encoder.') and p.requires_grad
        )
        head_train = sum(
            1 for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad
        )
        print(f"Trainable parameter tensors: encoder={enc_train}, head/MST={head_train}")

    # ---------------- Execute Training ----------------
    # Pass ckpt_path=None because we have already loaded the weights
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)

    wandb.finish(quiet=True)