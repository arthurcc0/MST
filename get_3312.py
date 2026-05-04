import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pathlib import Path

PATH = Path(r"\\rad-maid-004\D\PENN-MRI\penn-preprocessed\data\33123853")
out_dir = Path(r"D:\Users\UFPB\gabriel ayres\MST\out_sub")
images = PATH.glob("*.dcm")
for image in images:
    sitk_image = sitk.ReadImage(str(image))
    img_array = np.squeeze(sitk.GetArrayFromImage(sitk_image))
    png_filename = out_dir / f"{image.stem}.png"

    plt.imsave(png_filename, img_array, cmap='gray')
