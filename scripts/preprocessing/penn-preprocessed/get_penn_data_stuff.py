import torchio as tio
from pathlib import Path
import numpy as np

def get_spacing():
    path_root = Path(r'\\rad-maid-004\D\PENN-MRI') 
    path_root_out = path_root / 'preprocessed'
    path_root_out_data = path_root_out / 'data'

    path_patients = [p for p in path_root_out_data.iterdir() if p.is_dir()]
    
    spacings = []
    i = 0
    for path_dir in path_patients:
        if i < 700:
            pre_path = path_dir / 'pre.nii.gz'
            if pre_path.exists():
                img = tio.ScalarImage(pre_path)
                spacings.append(img.spacing)
            else:
                print(f"'Pre.nii.gz' not found in {path_dir}")
            i += 1
        else:
            break

    spacings = np.array(spacings)
    print(f"Found {len(spacings)} images.")
    if len(spacings) > 0:
        print(f"Mean spacing: {spacings.mean(axis=0)}")
        print(f"Std spacing: {spacings.std(axis=0)}")
        print(f"Min spacing: {spacings.min(axis=0)}")
        print(f"Max spacing: {spacings.max(axis=0)}")

if __name__ == "__main__":
    get_spacing()
