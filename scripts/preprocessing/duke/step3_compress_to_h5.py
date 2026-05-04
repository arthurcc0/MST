from pathlib import Path
import h5py
import numpy as np
import pydicom
from tqdm import tqdm
import concurrent.futures
import os
import glob

def process_patient(patient_dir):
    """
    Processes a single patient directory, reading all DICOM series.

    Args:
        patient_dir (Path): The path to the patient's directory.

    Returns:
        tuple: A tuple containing the patient ID and a dictionary of scan data,
               or None if processing fails.
    """
    patient_id = patient_dir.name
    scans_data = {}
    try:
        # Find all subdirectories containing DICOM files
        for study_dir in patient_dir.iterdir():
            if study_dir.is_dir():
                for series_dir in study_dir.iterdir():
                    if series_dir.is_dir():
                        # Look for DICOM files in this series directory
                        dicom_files = list(series_dir.glob('*.dcm')) + list(series_dir.glob('*'))
                        dicom_files = [f for f in dicom_files if f.is_file() and not f.suffix in ['.txt', '.xml', '.json']]
                        
                        if dicom_files:
                            series_name = f"{study_dir.name}_{series_dir.name}"
                            series_data = []
                            metadata = {}
                            
                            for dcm_file in sorted(dicom_files):
                                try:
                                    ds = pydicom.dcmread(dcm_file, force=True)
                                    if hasattr(ds, 'pixel_array'):
                                        series_data.append(ds.pixel_array)
                                        if not metadata:  # Store metadata from first file
                                            metadata = {
                                                'StudyDescription': getattr(ds, 'StudyDescription', ''),
                                                'SeriesDescription': getattr(ds, 'SeriesDescription', ''),
                                                'Modality': getattr(ds, 'Modality', ''),
                                                'PatientID': getattr(ds, 'PatientID', ''),
                                                'StudyDate': getattr(ds, 'StudyDate', ''),
                                                'SeriesNumber': getattr(ds, 'SeriesNumber', ''),
                                                'SliceThickness': getattr(ds, 'SliceThickness', ''),
                                                'PixelSpacing': getattr(ds, 'PixelSpacing', []).copy() if hasattr(ds, 'PixelSpacing') else []
                                            }
                                except Exception as e:
                                    print(f"Could not read DICOM file {dcm_file}: {e}")
                                    continue
                            
                            if series_data:
                                # Stack the series data into a 3D array
                                series_array = np.stack(series_data, axis=0)
                                scans_data[series_name] = {
                                    'data': series_array,
                                    'metadata': metadata
                                }
        
        return patient_id, scans_data if scans_data else None
    except Exception as e:
        print(f"Could not process directory {patient_dir}: {e}")
        return None

def compress_to_h5(path_root_in_data, path_h5_out, num_workers=4):
    """
    Compresses DICOM files into an HDF5 file using multiple threads.

    Args:
        path_root_in_data (Path): The root directory of the DICOM data.
        path_h5_out (Path): The path to the output HDF5 file.
        num_workers (int): The number of worker threads to use.
    """
    patient_dirs = sorted([p for p in path_root_in_data.iterdir() if p.is_dir()])

    with h5py.File(path_h5_out, 'w') as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all patient directories to the executor
            future_to_patient = {executor.submit(process_patient, p): p for p in patient_dirs}

            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_patient), total=len(patient_dirs), desc="Compressing to H5"):
                result = future.result()
                if result:
                    patient_id, scans_data = result
                    patient_group = f.create_group(patient_id)
                    for scan_name, scan_content in scans_data.items():
                        patient_group.create_dataset(scan_name, data=scan_content['data'], compression="gzip")
                        # Store metadata as attributes
                        metadata_group = patient_group.create_group(f"{scan_name}_metadata")
                        for key, value in scan_content['metadata'].items():
                            if isinstance(value, (str, int, float)):
                                metadata_group.attrs[key] = value
                            elif isinstance(value, list) and value:
                                metadata_group.attrs[key] = np.array(value)

if __name__ == "__main__":
    # Define the root path for the Duke-Cancer_MRI manifest data
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\manifest-1654812109500')

    # Input directory with raw DICOM data
    path_root_in_data = path_root / 'Duke-Breast-Cancer-MRI'

    # Output HDF5 file path
    path_h5_out = path_root / 'manifest_dicom_compressed.h5'

    # Set the number of worker threads (can be adjusted based on your system)
    # os.cpu_count() can be a good starting point
    num_workers = os.cpu_count() or 4

    # Create the output directory if it doesn't exist
    path_h5_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input data directory: {path_root_in_data}")
    print(f"Output HDF5 file: {path_h5_out}")
    print(f"Using {num_workers} worker threads.")

    # Run the compression function
    compress_to_h5(path_root_in_data, path_h5_out, num_workers=num_workers)

    print("Compression to HDF5 completed.")
