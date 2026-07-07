import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from 'mst'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import argparse
import pandas as pd
import torch
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import functools
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from monai.metrics import compute_average_surface_distance, compute_iou, compute_dice
from monai.networks.utils import one_hot
from mst.utils.roc_curve import plot_roc_curve

def safe_div(x, y):
    if y == 0:
        return float('nan')
    return x / y

def cm2acc(cm):
    """Calculates accuracy from a confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    return safe_div((tn + tp), (tn + tp + fn + fp))

def cm2x(cm):
    """Calculates sensitivity and specificity from a confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    p = tp + fn  # actual positive
    n = fp + tn  # actual negative
    sens = safe_div(tp, p)  # sensitivity
    spec = safe_div(tn, n)  # specificity
    return None, None, sens, spec  # Return format matches unpacking in plot_confusion_matrix

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(df, output_dir):
    """Plots and saves a voxel-level confusion matrix in the style of main_predict.py."""
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}
    # Sum TP, FP, FN, TN from the dataframe
    tp = df['tp'].sum()
    fp = df['fp'].sum()
    fn = df['fn'].sum()
    tn = df['tn'].sum()

    # The confusion matrix array in [[TN, FP], [FN, TP]] format
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Calculate metrics
    acc = cm2acc(cm)
    _, _, sens, spec = cm2x(cm)
    
    df_cm = pd.DataFrame(data=cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm, ax=ax, cbar=False, fmt='d', annot=True, cmap='Blues')
    
    title = (f'Voxel-Level Confusion Matrix\n'
             f'ACC={acc:.3f} | Sens={sens:.3f} | Spec={spec:.3f}')
    ax.set_title(title, fontdict=fontdict)
    ax.set_xlabel('Prediction', fontdict=fontdict)
    ax.set_ylabel('True', fontdict=fontdict)
    
    plt.tight_layout()
    plot_path = output_dir / 'confusion_matrix.png'
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Saved confusion matrix to {plot_path}")
    plt.close(fig)

def calculate_all_metrics(pred_img: sitk.Image, true_img: sitk.Image):
    """Calculates a suite of segmentation metrics using MONAI for binary masks."""
    spacing = pred_img.GetSpacing()
    voxel_volume = np.prod(spacing)

    # Get numpy arrays
    pred_data = sitk.GetArrayFromImage(pred_img)
    true_data = sitk.GetArrayFromImage(true_img)

    # Convert to binary torch tensors and add batch & channel dimensions
    pred_tensor = torch.tensor(pred_data > 0, dtype=torch.int).unsqueeze(0).unsqueeze(0)
    true_tensor = torch.tensor(true_data > 0, dtype=torch.int).unsqueeze(0).unsqueeze(0)
    
    # One-hot encode the tensors
    pred_one_hot = one_hot(pred_tensor, num_classes=2)
    true_one_hot = one_hot(true_tensor, num_classes=2)

    gt_voxel_count = (true_data > 0).sum().item()

    if gt_voxel_count == 0:
        is_empty_pred = (pred_data > 0).sum().item() == 0
        dice_val = 1.0 if is_empty_pred else 0.0
        return {
            'dice': dice_val,
            'jaccard': dice_val,
            'assd': 0.0 if is_empty_pred else np.nan,
            'dice_foreground': dice_val,
            'jaccard_foreground': dice_val,
            'assd_foreground': 0.0 if is_empty_pred else np.nan,
            'gt_voxel_count': 0,
            'gt_volume_mm3': 0.0,
            'tp': 0,
            'fp': (pred_data > 0).sum().item(),
            'fn': 0,
            'tn': np.size(true_data) - (pred_data > 0).sum().item(),
            'pred_voxel_count': (pred_data > 0).sum().item()
        }

    # Calculate metrics for both classes
    dice_all = compute_dice(y_pred=pred_one_hot, y=true_one_hot, include_background=True)
    jaccard_all = compute_iou(y_pred=pred_one_hot, y=true_one_hot, include_background=True)
    assd_all = compute_average_surface_distance(y_pred=pred_one_hot, y=true_one_hot, include_background=True, spacing=spacing)

    # Voxel-level TP, FP, FN
    pred_flat = (pred_data > 0).flatten()
    true_flat = (true_data > 0).flatten()
    
    tp = np.sum(np.logical_and(pred_flat, true_flat))
    fp = np.sum(np.logical_and(pred_flat, np.logical_not(true_flat)))
    fn = np.sum(np.logical_and(np.logical_not(pred_flat), true_flat))
    tn = np.sum(np.logical_and(np.logical_not(pred_flat), np.logical_not(true_flat)))
    pred_voxel_count = (pred_data > 0).sum().item()

    return {
        'dice': dice_all.mean().item(),
        'jaccard': jaccard_all.mean().item(),
        'assd': assd_all.mean().item(),
        'dice_foreground': dice_all[0, 1].item(),
        'jaccard_foreground': jaccard_all[0, 1].item(),
        'assd_foreground': assd_all[0, 1].item(),
        'gt_voxel_count': gt_voxel_count,
        'gt_volume_mm3': gt_voxel_count * voxel_volume,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'pred_voxel_count': pred_voxel_count
    }

def process_patient(patient_filename, pred_dir, true_dir, debug_out_dir=None):
    try:
        # patient_filename is like 'Breast_MRI_001_reconstructed.nii.gz'
        patient_id_str = patient_filename.replace('_reconstructed.nii.gz', '')

        # Extract only the numeric part of the patient ID
        patient_id_numeric_str = ''.join(filter(str.isdigit, patient_id_str))
        if not patient_id_numeric_str:
            logging.warning(f"Could not extract numeric patient ID from {patient_filename}. Skipping.")
            return patient_filename, {'error': 'Could not extract numeric patient ID'}
        
        # Find the corresponding ground truth file
        true_path = None
        logging.info(f"Attempting to match '{patient_filename}' with numeric ID '{patient_id_numeric_str}'")
        for prefix in ["duke_", "ispy1_", "ispy2_"]:
            potential_path = true_dir / f"{prefix}{patient_id_numeric_str}.nii.gz"
            logging.debug(f"Checking for ground truth file at: {potential_path}")
            if potential_path.exists():
                true_path = potential_path
                logging.info(f"Found ground truth file: {true_path}")
                break

        if not true_path:
            logging.warning(f"Ground truth file not found for patient ID {patient_id_numeric_str}")
            return patient_id_str, {'error': 'Ground truth file not found'}

        pred_path = pred_dir / patient_filename
        if not pred_path.exists():
            logging.error(f"Prediction file not found: {pred_path}")
            return patient_id_str, {'error': 'Prediction file not found'}

        pred_img = sitk.ReadImage(str(pred_path))
        true_img = sitk.ReadImage(str(true_path))

        # Align the prediction to the ground truth's space
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(true_img)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        pred_aligned = resample.Execute(pred_img)

        if debug_out_dir:
            debug_out_dir.mkdir(exist_ok=True)
            sitk.WriteImage(pred_img, str(debug_out_dir / f"{patient_id_str}_pred_original.nii.gz"))
            sitk.WriteImage(true_img, str(debug_out_dir / f"{patient_id_str}_true_original.nii.gz"))
            sitk.WriteImage(pred_aligned, str(debug_out_dir / f"{patient_id_str}_pred_aligned.nii.gz"))
            logging.info(f"Saved debug images for patient {patient_id_str} to {debug_out_dir}")

        # Calculate metrics between the aligned prediction and the original ground truth
        metrics = calculate_all_metrics(pred_aligned, true_img)
        return patient_id_str, metrics

    except Exception as e:
        logging.error(f"Error processing patient {patient_filename}: {e}")
        return patient_filename, {'error': str(e)}

def plot_metrics(foreground_df, output_dir):
    """Generates and saves histograms for the evaluation metrics."""
    output_dir.mkdir(exist_ok=True)
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Segmentation Metric Distributions (Foreground Cases)', fontsize=16, fontweight='bold')

    # Plot Dice
    sns.histplot(foreground_df['dice_foreground'], bins=20, ax=axes[0], kde=True)
    axes[0].set_title('Dice Coefficient (Foreground)', fontdict=fontdict)
    axes[0].set_xlabel('Dice Score')
    axes[0].set_ylabel('Frequency')

    # Plot Jaccard
    sns.histplot(foreground_df['jaccard_foreground'], bins=20, ax=axes[1], kde=True)
    axes[1].set_title('Jaccard Index (IoU) (Foreground)', fontdict=fontdict)
    axes[1].set_xlabel('Jaccard Score')

    # Plot ASSD
    assd_valid = foreground_df['assd_foreground'].dropna()
    if not assd_valid.empty:
        sns.histplot(assd_valid, bins=20, ax=axes[2], kde=True)
    axes[2].set_title('Average Surface Distance (ASSD) (Foreground)', fontdict=fontdict)
    axes[2].set_xlabel('ASSD (mm)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = output_dir / 'metrics_distribution.png'
    fig.savefig(save_path, dpi=300)
    logging.info(f"Saved metric distribution plot to {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructed segmentations against ground truth.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--reconstructed_dir', type=str, required=True, help='Directory containing the reconstructed segmentation files.')
    parser.add_argument('--ground_truth_dir', type=str, required=True, help='Directory containing the ground truth segmentation files.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save the output CSV and plots.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for processing.')
    parser.add_argument('--debug_patient_id', type=str, default=None, help='Run only for a single patient ID and save intermediate images for debugging.')

    args = parser.parse_args()

    pred_dir = Path(args.reconstructed_dir)
    true_dir = Path(args.ground_truth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.debug_patient_id:
        patient_id = args.debug_patient_id
        debug_out_dir = Path(f"debug_output_{patient_id}")
        logging.info(f"--- Running in DEBUG mode for patient {patient_id} ---")
        _, metrics = process_patient(patient_id, pred_dir, true_dir, debug_out_dir)
        logging.info(f"Debug metrics: {metrics}")
        return # Exit after debug run

    # Find all segmentation files in the reconstructed directory
    patient_files = sorted(pred_dir.glob('*_reconstructed.nii.gz'))
    if not patient_files:
        logging.error(f"No segmentation files found in {pred_dir} with pattern '*_reconstructed.nii.gz'")
        return
    patient_ids = sorted([p.name for p in patient_files])

    logging.info(f"Found {len(patient_ids)} patients to evaluate.")
    if not patient_ids:
        logging.warning("No patients found to evaluate.")
        return

    process_func = functools.partial(process_patient, pred_dir=pred_dir, true_dir=true_dir)

    all_metrics = []
    with Pool(args.num_workers) as pool:
        for patient_id, metrics in tqdm(pool.imap_unordered(process_func, patient_ids), total=len(patient_ids)):
            if 'error' not in metrics:
                metrics['patient_id'] = patient_id
                all_metrics.append(metrics)
            else:
                logging.error(f"Skipping patient {patient_id} due to error: {metrics['error']}")

    if not all_metrics:
        logging.warning("No valid results were generated.")
        return

    df = pd.DataFrame(all_metrics)
    cols = ['patient_id', 'dice', 'jaccard', 'assd', 'dice_foreground', 'jaccard_foreground', 'assd_foreground', 'gt_voxel_count', 'gt_volume_mm3', 'tp', 'fp', 'fn', 'tn', 'pred_voxel_count']
    df = df[cols]
    output_csv_path = output_dir / 'evaluation_results.csv'
    df.to_csv(output_csv_path, index=False)

    logging.info(f"\nEvaluation complete. Results saved to {output_csv_path}")

    # --- Summary Statistics ---
    foreground_df = df[df['gt_voxel_count'] > 0].copy()


    logging.info(f"Evaluation complete. Results saved in {output_dir}")

    if not foreground_df.empty:
        logging.info("\n--- Foreground Statistics ---")
        logging.info(f"Dice Foreground: {foreground_df['dice_foreground'].mean():.3f} ± {foreground_df['dice_foreground'].std():.3f}")
        logging.info(f"Jaccard Foreground: {foreground_df['jaccard_foreground'].mean():.3f} ± {foreground_df['jaccard_foreground'].std():.3f}")
        logging.info(f"ASSD Foreground: {foreground_df['assd_foreground'].mean():.2f} ± {foreground_df['assd_foreground'].std():.2f}")
        logging.info(f"GT Volume (mm^3): {foreground_df['gt_volume_mm3'].mean():.2f} ± {foreground_df['gt_volume_mm3'].std():.2f}")
    else:
        logging.info("No foreground cases found to calculate summary statistics.")

    # --- Plotting ---
    if not foreground_df.empty:
        plot_metrics(foreground_df, output_dir)
        plot_confusion_matrix(foreground_df, output_dir)

    # --- ROC-AUC Curve ---
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}
    y_true = (df['gt_voxel_count'] > 0).astype(int)
    y_pred_scores = df['pred_voxel_count'].to_numpy()

    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true, y_pred_scores, axis, fontdict=fontdict)
    fig.tight_layout()
    roc_path = output_dir / 'roc_curve.png'
    fig.savefig(roc_path, dpi=300)
    logging.info(f"Saved ROC curve to {roc_path} with AUC = {auc_val:.3f}")
    plt.close(fig)

if __name__ == "__main__":
    main()
