import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import zlib

path_root = Path(r'\\rad-maid-004\D\PENN-MRI') 
path_root_out = path_root/'pennmasked'
path_root_out_data = path_root_out/'data'

def rotate90_clockwise(path_file):
    """
    Rotates a NIfTI image 90 degrees clockwise and saves it.
    """
    path_file_out = path_file
    try:
        nii_img = nib.load(path_file)
        img_data = nii_img.get_fdata()
    except (nib.filebasedimages.ImageFileError, RuntimeError, EOFError, zlib.error) as e:
        # Return the error to be handled in the main process if needed
        return f"Skipping file {path_file} due to error: {e}"

    rotated_data = np.rot90(img_data, k=-1, axes=(0, 1))
    new_nii_img = nib.Nifti1Image(rotated_data, nii_img.affine, nii_img.header)
    path_file_out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_nii_img, path_file_out)
    return f'Rotated and saved to {path_file_out}'

if __name__ == '__main__':
    source_data_path = path_root_out_data
    files_to_process = []
    for file_path in source_data_path.rglob('*.nii.gz'):
        if 'pre' in file_path.name or 'post' in file_path.name:
            files_to_process.append(file_path)

    if not files_to_process:
        print(f"No 'pre' or 'post' .nii.gz files found in {source_data_path}")
    else:
        print(f"Found {len(files_to_process)} files to rotate.")

        with Pool(processes=8) as pool:
            with tqdm(total=len(files_to_process)) as pbar:
                for result in pool.imap_unordered(rotate90_clockwise, files_to_process):
                    if 'Skipping' in result:
                        print(result) # Print errors
                    pbar.update()
