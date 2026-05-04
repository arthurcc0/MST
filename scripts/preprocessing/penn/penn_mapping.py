
import os
import pandas as pd
from tqdm import tqdm

base_dir = '\\\\10.156.155.77\mccarthy_lab\MRI'

def create_mapping():
    """
    Creates a long-format CSV mapping Patient IDs to their Pre and Post contrast
    MRI file paths.
    """
    records = []
    data_dir = os.path.join(base_dir, 'data')

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    for patient_entry in tqdm(os.scandir(data_dir), desc="Processing patients"):
        if not patient_entry.is_dir():
            continue
        
        patient_id = patient_entry.name.lower()
        
   
        pre_path = os.path.join(patient_entry.path, 'Pre')
        if os.path.isdir(pre_path):
            for file_entry in os.scandir(pre_path):
                if file_entry.is_file():
                    full_path = file_entry.path.replace('\\', '/')
                    records.append({
                        'PatientID': patient_id,
                        'Contrast': 'Pre',
                        'FileName': file_entry.name,
                        'FullPath': full_path
                    })

        post_path = os.path.join(patient_entry.path, 'Post')
        if os.path.isdir(post_path):
            for file_entry in os.scandir(post_path):
                if file_entry.is_file():
                    full_path = file_entry.path.replace('\\', '/')
                    records.append({
                        'PatientID': patient_id,
                        'Contrast': 'Post',
                        'FileName': file_entry.name,
                        'FullPath': full_path
                    })

    if records:
        df = pd.DataFrame(records)
        output_path = 'penn_mapping.csv'
        df.to_csv(output_path, index=False)
        print(f"Mapping created successfully at {output_path}")
    else:
        print("No files found to create a mapping.")


if __name__ == "__main__":
    create_mapping()
