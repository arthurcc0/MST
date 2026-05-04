from pathlib import Path
import h5py
import numpy as np
import torchio as tio
from tqdm import tqdm

def decompress_from_h5(path_h5_in, path_root_out_data):
    """
    Decompresses images from an HDF5 file and saves them as NIfTI files,
    recreating the original directory structure.

    Args:
        path_h5_in (Path): The path to the input HDF5 file.
        path_root_out_data (Path): The root directory to save the decompressed data.
    """
    with h5py.File(path_h5_in, 'r') as f:
        patient_ids = list(f.keys())
        for patient_id in tqdm(patient_ids, desc="Decompressing from H5"):
            patient_group = f[patient_id]
            patient_dir_out = path_root_out_data / patient_id
            patient_dir_out.mkdir(parents=True, exist_ok=True)

            scan_names = [key for key in patient_group.keys() if not key.endswith('_affine')]

            for scan_name in scan_names:
                try:
                    data = patient_group[scan_name][()]
                    affine = patient_group[f"{scan_name}_affine"][()]

                    # Create a torchio ScalarImage with the data and affine matrix
                    image = tio.ScalarImage(tensor=data, affine=affine)

                    # Define the output path for the NIfTI file
                    nii_out_path = patient_dir_out / f"{scan_name}.nii.gz"

                    # Save the image as a NIfTI file
                    image.save(nii_out_path)
                except KeyError as e:
                    print(f"Skipping scan in {patient_id} due to missing data/affine: {e}")
                except Exception as e:
                    print(f"Could not process {scan_name} for patient {patient_id}: {e}")

if __name__ == "__main__":
    # Define the root path for the Duke-Cancer_MRI data
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI')

    # Input HDF5 file path
    path_h5_in = path_root /'data_compressed.h5'

    # Output directory for the decompressed data
    path_root_out_data = path_root / 'preprocessed_crop-with-clip' / 'data_decompressed'

    # Create the output directory if it doesn't exist
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    print(f"Input HDF5 file: {path_h5_in}")
    print(f"Output data directory: {path_root_out_data}")

    # Run the decompression function
    decompress_from_h5(path_h5_in, path_root_out_data)

    print("Decompression from HDF5 completed.")