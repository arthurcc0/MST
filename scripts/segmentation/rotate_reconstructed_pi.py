import SimpleITK as sitk 
import os 
import numpy as np
from pathlib import Path
PATH_RECONSTRUCTED = Path(r"\\rad-maid-004\D\Duke-Cancer_MRI\reconstructed_segmentations")

def rotate_180_reconstructed_pi(path_img):
    for img in path_img.glob('*.nii.gz'):
        n_img = sitk.ReadImage(img)
        img_array = sitk.GetArrayFromImage(n_img)
        img_array = np.rot90(img_array, k=2, axes=(0, 1))
        n_img = sitk.GetImageFromArray(img_array)
        n_img.CopyInformation(n_img)
        sitk.WriteImage(n_img, img)

def rotate_z_axis(path_img):
    for img in path_img.glob("*.nii.gz"):
        n_img = sitk.ReadImage(img)
        n_img = sitk.Flip(n_img, [False, False, True])
        n_img.CopyInformation(n_img)
        sitk.WriteImage(n_img, img)

def rotate_y_axis(path_img):
    for img in path_img.glob("*.nii.gz"):
        n_img = sitk.ReadImage(img)
        n_img = sitk.Flip(n_img, [False, True, False])
        n_img.CopyInformation(n_img)
        sitk.WriteImage(n_img, img)


def rotate_x_axis(path_img):
    for img in path_img.glob("*.nii.gz"):
        n_img = sitk.ReadImage(img)
        n_img = sitk.Flip(n_img, [True, False, False])
        n_img.CopyInformation(n_img)
        sitk.WriteImage(n_img, img)

if __name__ == "__main__":
    #rotate_180_reconstructed_pi(PATH_RECONSTRUCTED)
    #rotate_z_axis(PATH_RECONSTRUCTED)
    rotate_x_axis(PATH_RECONSTRUCTED)