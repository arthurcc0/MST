from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import numpy as np
import os
import nibabel as nib
import torch
from monai.metrics import compute_dice
from monai.transforms import AsDiscrete
import pandas as pd
from tqdm import tqdm

mammamia_path = Path(r"\\10.156.155.77\maidment_lab\bbruno\Mama-MIA\segmentations\expert")
reconstructed_path = Path(r"\\rad-maid-004\D\Duke-Cancer_MRI\reconstructed_segmentations")

def slice_histogram(img_array, iter: int):
    std_path = Path('./')
    
    os.makedirs(std_path/'histograms', exist_ok=True)
    
    min_val = 0
    max_val = 1000
    print(f"Slice {iter} range: {min_val} to {max_val}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(img_array.flatten(), kde=True, bins=100)
    
    plt.title(f"Histogram of Slice {iter} (Range: {min_val}-{max_val})")
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    
    plt.savefig(str(std_path/'histograms'/f'histo_{iter}.png'))
    plt.close()

def subtraction():
    std_path = Path('./')
    path_d0 = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed-mst-with-seg\data\Breast_MRI_001')
    path_d1 = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed-mst-with-seg\data\Breast_MRI_001')

    pre_img = sitk.ReadImage(str(path_d0/'pre.nii.gz'))
    post_img = sitk.ReadImage(str(path_d1/'post_1.nii.gz'))
    
    pre_arr = sitk.GetArrayFromImage(pre_img)
    post_arr = sitk.GetArrayFromImage(post_img)
    
    sub_arr = post_arr - pre_arr
    sub_arr = sub_arr - sub_arr.min()
    sub_arr = sub_arr.astype(np.uint16)
    sub_i2 = sitk.GetImageFromArray(sub_arr)
    # for i in range(sub_i.shape[0]):
    #     slice_histogram(sub_i[:,:,i], i)
    #     print("slice {} done!".format(i))
   
    sitk.WriteImage(sub_i2, str(std_path/'sub.nii.gz'))

# def dice_testing(mammamia_path, reconstructed_path):
#     """Calculates the Dice score between two segmentation images."""
#     try:
#         # Load images
#         mama_img = sitk.ReadImage(str(mammamia_path))
#         recon_img = sitk.ReadImage(str(reconstructed_path))

#         # Get numpy arrays
#         m_nimg = sitk.GetArrayFromImage(mama_img)
#         r_nimg = sitk.GetArrayFromImage(recon_img)

#         # Convert to binary torch tensors and add batch & channel dimensions
#         m_tensor = torch.tensor(m_nimg > 0, dtype=torch.int).unsqueeze(0).unsqueeze(0)
#         r_tensor = torch.tensor(r_nimg > 0, dtype=torch.int).unsqueeze(0).unsqueeze(0)
        
#         # One-hot encode the tensors
#         to_one_hot = AsDiscrete(to_onehot=2)
#         m_one_hot = to_one_hot(m_tensor)
#         r_one_hot = to_one_hot(r_tensor)
        
#         # Compute dice. y_pred is the prediction, y is the ground truth.
#         dice = compute_dice(y_pred=r_one_hot, y=m_one_hot, include_background=False)
#         return dice.mean().item()
#     except Exception as e:
#         # print(f"Error processing {reconstructed_path.name}: {e}")
#         return None


# if __name__ == "__main__":
#     reconstructed_dir = Path(r"\\rad-maid-004\D\Duke-Cancer_MRI\reconstructed_segmentations")
#     mammamia_dir = Path(r"\\10.156.155.77\maidment_lab\bbruno\Mama-MIA\segmentations\expert")
    
#     results = []
    
#     reconstructed_files = sorted(list(reconstructed_dir.glob('Breast_MRI_*_reconstructed.nii.gz')))
    
#     for recon_path in tqdm(reconstructed_files, desc="Comparing Segmentations"):
#         try:
#             patient_id = recon_path.name.split('_')[2]
#         except IndexError:
#             # print(f"Could not parse patient ID from {recon_path.name}")
#             continue

#         # Check for different possible ground truth prefixes
#         mammamia_path = None
#         for prefix in ["duke_", "ispy1_", "ispy2_"]:
#              potential_path = mammamia_dir / f"{prefix}{patient_id}.nii.gz"
#              if potential_path.exists():
#                  mammamia_path = potential_path
#                  break
        
#         if mammamia_path is None:
#             continue

#         dice_score = dice_testing(mammamia_path, recon_path)
        
#         if dice_score is not None:
#             results.append({'patient_id': patient_id, 'dice': dice_score})

#     if results:
#         df = pd.DataFrame(results)
#         df['patient_id'] = df['patient_id'].astype(int)
#         df.sort_values(by='patient_id', inplace=True)
#         df.reset_index(drop=True, inplace=True)
        
#         print("\n--- Evaluation Summary ---")
#         print(df.to_string())
#         print(f"\nMean Dice: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
        
#         output_csv = Path('./mammamia_comparison_results.csv')
#         df.to_csv(output_csv, index=False)
#         print(f"\nResults saved to {output_csv}")
#     else:
#         print("No matching files found to compare.")
if __name__ == "__main__":
    subtraction()