from pathlib import Path
import logging
import pandas as pd
import numpy as np
import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk
import sys

# Setup basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        return None  # Don't store this type of data
    elif isinstance(x, pydicom.dataset.Dataset):
        return None  # Don't store this type of data
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x

def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword: value for key in ds.keys()
            if ((keyword := ds[key].keyword) not in exclude) and ((value := maybe_convert(ds[key].value)) is not None)}

def series2nifti_for_debug(series_info_tuple, path_root_in_str, path_root_out_data_str):
    target_output_patient_folder, seq_name, classic_path_parent_str = series_info_tuple
    
    path_root_in = Path(path_root_in_str)
    path_root_out_data = Path(path_root_out_data_str)
    
    # This is the path to the directory containing DICOM files for the series
    path_series_absolute = path_root_in / Path(classic_path_parent_str)

    logger.info(f"Processing series: {seq_name} for patient folder {target_output_patient_folder}")
    logger.info(f"Input DICOM directory: {path_series_absolute}")

    if not path_series_absolute.is_dir():
        logger.warning(f"Expected directory but found file or non-existent: {path_series_absolute}")
        return None

    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = list(path_series_absolute.glob('*.dcm'))
        if not dicom_files:
            logger.warning(f"No .dcm files found in {path_series_absolute}")
            return None

        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series_absolute))
        if not dicom_names:
            logger.warning(f"SimpleITK could not find DICOM series in {path_series_absolute}")
            return None
            
        reader.SetFileNames(dicom_names)
        img_nii = reader.Execute()

        # Define output directory using the target_output_patient_folder
        path_out_dir = path_root_out_data / target_output_patient_folder
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write NIfTI file
        path_file = path_out_dir / f'{seq_name}.nii.gz'
        logger.info(f"Writing NIfTI file to: {path_file}")
        sitk.WriteImage(img_nii, str(path_file)) # Ensure path is string for sitk

        # Optionally, read and return metadata if needed later
        # first_dcm_file = dicom_files[0]
        # ds = pydicom.dcmread(first_dcm_file, stop_before_pixels=True)
        # metadata = dataset2dict(ds)
        # metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        # return metadata
        return str(path_file) # Return path of created file for confirmation

    except Exception as e:
        logger.error(f"Error processing series '{seq_name}' from DICOMs in: {path_series_absolute}")
        logger.error(str(e))
        return None

if __name__ == "__main__":
    # Path settings
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI')
    path_root_in_dicoms = path_root / 'manifest-1654812109500' # Root for input DICOMs based on classic_path
    path_root_out_nifti = path_root / 'preprocessed-mst-with-seg' / 'data'
    path_root_out_nifti.mkdir(parents=True, exist_ok=True)

    # Target patient information
    target_patient_id_numeric = 922
    # This is the crucial part: the output folder name step2a_calc_sub.py expects
    target_output_patient_folder_name = f"Breast_MRI_{target_patient_id_numeric}" 

    logger.info(f"Targeting Patient ID (numeric): {target_patient_id_numeric}")
    logger.info(f"Target NIfTI output folder name: {target_output_patient_folder_name}")
    logger.info(f"NIfTI output base directory: {path_root_out_nifti}")
    logger.info(f"DICOM input base directory (for classic_path): {path_root_in_dicoms}")

    # Load mapping files
    try:
        df_filepath_mapping = pd.read_excel(path_root / 'Breast-Cancer-MRI-filepath_filename-mapping.xlsx')
        df_my_mapping = pd.read_excel(path_root / 'my-mapping.xlsx')
    except FileNotFoundError as e:
        logger.error(f"Error loading mapping files: {e}. Make sure they are in {path_root}")
        sys.exit(1)

    # Prepare DataFrame similar to step1_dicom2nifti.py
    df_combined = df_filepath_mapping[df_filepath_mapping.columns[:4]].copy()
    seq_paths_split = df_combined['original_path_and_filename'].str.split('/')
    
    # Extract numeric PatientID from 'original_path_and_filename' (e.g., 'DICOM_Images/Breast_MRI_922/... -> 922)
    df_combined['PatientID_num_orig'] = seq_paths_split.apply(lambda x: int(x[1].rsplit('_', 1)[1]) if len(x) > 1 and '_' in x[1] else None)
    df_combined['SequenceName_orig'] = seq_paths_split.apply(lambda x: x[2] if len(x) > 2 else None)
    
    # Add 'classic_path_parent' from my-mapping.xlsx
    # 'classic_path' in my-mapping.xlsx is the full path to a DICOM file.
    # We need its parent directory for SimpleITK's series reader.
    # The 'classic_path' values are like 'Duke-Breast-Cancer-MRI/BreastMRIXXX/...' relative to 'manifest-1654812109500'
    df_combined['classic_path_parent'] = df_my_mapping['classic_path'].apply(lambda p: str(Path(p).parent))

    # Filter for the target patient
    df_patient = df_combined[df_combined['PatientID_num_orig'] == target_patient_id_numeric].copy()

    if df_patient.empty:
        logger.error(f"No series found for Patient ID {target_patient_id_numeric} in the mapping files.")
        sys.exit(1)

    # Drop duplicates based on the combination that defines a unique series for NIfTI conversion
    # For a given patient, each sequence (e.g., 'post_1') should result in one NIfTI file.
    # 'classic_path_parent' points to the DICOMs for that sequence.
    df_patient = df_patient.drop_duplicates(subset=['SequenceName_orig', 'classic_path_parent'], keep='first')
    logger.info(f"Found {len(df_patient)} unique series to process for patient {target_patient_id_numeric}.")

    series_to_process_for_patient922 = []
    for _, row in df_patient.iterrows():
        # series_info_tuple: (target_output_folder, seq_name, path_to_dicom_folder_relative_to_path_root_in_dicoms)
        series_info = (
            target_output_patient_folder_name, 
            row['SequenceName_orig'], 
            row['classic_path_parent']
        )
        series_to_process_for_patient922.append(series_info)

    processed_files_count = 0
    for series_data in tqdm(series_to_process_for_patient922, desc=f"Processing Patient {target_patient_id_numeric}"):
        logger.info(f"Calling series2nifti_for_debug with: {series_data}")
        result_path = series2nifti_for_debug(series_data, str(path_root_in_dicoms), str(path_root_out_nifti))
        if result_path:
            logger.info(f"Successfully created: {result_path}")
            processed_files_count += 1
        else:
            logger.warning(f"Failed to process series: {series_data[1]} from {series_data[2]}")

    logger.info(f"Finished processing. Created {processed_files_count} NIfTI files for patient {target_output_patient_folder_name}.")

    # Verify if the specific problematic file was created:
    expected_file = path_root_out_nifti / target_output_patient_folder_name / 'post_1.nii.gz'
    if expected_file.exists():
        logger.info(f"SUCCESS: The target file {expected_file} was created!")
    else:
        logger.error(f"FAILURE: The target file {expected_file} was NOT created.")

