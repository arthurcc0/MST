#!/usr/bin/env python
"""
Script to generate a filepath-filename mapping for Duke Breast Cancer MRI data
with the new directory structure.
"""

import os
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm
import glob

# Base directory for the Duke Breast Cancer MRI dataset
base_dir = r'\\rad-maid-004\D\Duke-Cancer_MRI\manifest-1654812109500\Duke-Breast-Cancer-MRI'

def create_mapping():
    """Create mapping dataframe from the directory structure."""
    print(f"Scanning directory: {base_dir}")
    
    # Lists to store data
    patient_ids = []
    sequence_names = []
    original_paths = []
    classic_paths = []
    filenames = []
    
    # Pattern to extract patient ID from directory name (e.g. Breast_MRI_001)
    patient_id_pattern = re.compile(r'Breast_MRI_(\d+)')
    
    # Find all patient directories
    patient_dirs = glob.glob(os.path.join(base_dir, "Breast_MRI_*"))
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        # Extract patient ID
        patient_dir_name = os.path.basename(patient_dir)
        patient_id_match = patient_id_pattern.match(patient_dir_name)
        if not patient_id_match:
            print(f"Skipping directory with invalid format: {patient_dir}")
            continue
        
        patient_id = patient_id_match.group(1)
        
        # Find all study directories for this patient
        study_dirs = glob.glob(os.path.join(patient_dir, "*"))
        
        for study_dir in study_dirs:
            # Find all series directories for this study
            series_dirs = glob.glob(os.path.join(study_dir, "*"))
            
            for series_dir in series_dirs:
                series_dir_name = os.path.basename(series_dir)
                
                # Extract sequence name from series directory (e.g. "3.000000-ax dyn pre-93877")
                # Format appears to be: [number]-[sequence description]-[id]
                parts = series_dir_name.split('-', 1)
                if len(parts) >= 2:
                    sequence_name = parts[1].rsplit('-', 1)[0].strip()  # Extract middle part as sequence name
                else:
                    sequence_name = series_dir_name
                
                # Check for DICOM files
                dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
                if not dicom_files:
                    # If no direct .dcm files, check subdirectories
                    subdirs = glob.glob(os.path.join(series_dir, "*"))
                    for subdir in subdirs:
                        if os.path.isdir(subdir):
                            dicom_files.extend(glob.glob(os.path.join(subdir, "*.dcm")))
                
                # If no DICOM files found, skip this series
                if not dicom_files:
                    print(f"No DICOM files found in {series_dir}")
                    continue
                
                # For each DICOM file found, add an entry
                for dcm_file in dicom_files:
                    relative_path = os.path.relpath(dcm_file, base_dir)
                    original_path = '/'.join(relative_path.split(os.sep))  # Convert to forward slashes
                    
                    patient_ids.append(int(patient_id))
                    sequence_names.append(sequence_name)
                    original_paths.append(f"Duke-Breast-Cancer-MRI/{original_path}")  # Format as per original mapping
                    classic_paths.append(dcm_file)
                    filenames.append(os.path.basename(dcm_file))

    
        # Find segmentation files
    seg_base_dir = r'\\rad-maid-004\D\Duke-Cancer_MRI\segmentations\manifest-1654811613950\Duke-Breast-Cancer-MRI'
    if os.path.exists(seg_base_dir):
        print(f"Searching for segmentations in: {seg_base_dir}")
        seg_patient_dirs = glob.glob(os.path.join(seg_base_dir, "Breast_MRI_*"))

        for patient_dir in seg_patient_dirs:
            patient_dir_name = os.path.basename(patient_dir)
            patient_id_match = patient_id_pattern.match(patient_dir_name)
            if not patient_id_match:
                continue
            
            patient_id = patient_id_match.group(1)
            
            study_dirs = glob.glob(os.path.join(patient_dir, "*"))
            
            for study_dir in study_dirs:
                series_dirs = glob.glob(os.path.join(study_dir, "*"))
                
                for series_dir in series_dirs:
                    if 'Segmentation' in os.path.basename(series_dir):
                        dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
                        
                        for dcm_file in dicom_files:
                            relative_path = os.path.relpath(dcm_file, seg_base_dir)
                            original_path = '/'.join(relative_path.split(os.sep))
                            
                            patient_ids.append(int(patient_id))
                            sequence_names.append('Segmentation')
                            original_paths.append(f"Duke-Breast-Cancer-MRI/{original_path}")
                            classic_paths.append(dcm_file)
                            filenames.append(os.path.basename(dcm_file))
    else:
        print(f"Segmentation directory not found: {seg_base_dir}")
    
    
    # Create DataFrame
    df = pd.DataFrame({
        'classic_path': classic_paths,
        'original_path_and_filename': original_paths,
        'actual_filename': filenames,
        'file_exists': [True] * len(classic_paths),  # Assuming all files exist since we found them
        'PatientID': patient_ids,
        'SequenceName': sequence_names
    })
    
    return df

if __name__ == "__main__":
    # Create mapping
    df_mapping = create_mapping()
    
    # Save to Excel
    output_file = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\my-mapping.xlsx')
    print(f"Saving mapping to {output_file}")
    df_mapping.to_excel(output_file, index=False)
    
    # Also save as CSV for easier processing
    csv_output = output_file.with_suffix('.csv')
    df_mapping.to_csv(csv_output, index=False)
    
    print(f"Completed. Found {len(df_mapping)} DICOM files across {df_mapping['PatientID'].nunique()} patients.")
