def fix_mask_location(path_masks, path_data):
    """
    Replace mask.nii.gz files in path_data with corresponding mask files from path_masks.
    
    Args:
        path_masks (Path): Path to directory containing source mask files
        path_data (Path): Path to directory containing target directories with mask.nii.gz to replace
    """
    processed_count = 0
    error_count = 0
    
    # Iterate through all subdirectories in path_data
    for data_subdir in path_data.iterdir():
        if not data_subdir.is_dir():
            continue
            
        # Extract the directory name (e.g., "Breast_MRI_001_left")
        dir_name = data_subdir.name
        
        # Extract the base name without _left or _right suffix
        # e.g., "Breast_MRI_001_left" -> "Breast_MRI_001"
        if dir_name.endswith('_left') or dir_name.endswith('_right'):
            base_name = dir_name.rsplit('_', 1)[0]  # Remove last part after underscore
        else:
            print(f"Warning: Directory name {dir_name} doesn't follow expected pattern")
            error_count += 1
            continue
        
        # Check if corresponding directory exists in path_masks
        mask_subdir = path_masks / dir_name
        if not mask_subdir.exists():
            print(f"Warning: No corresponding mask directory found for {dir_name}")
            error_count += 1
            continue
            
        # Look for the specific mask file that matches the base name
        # e.g., look for "Breast_MRI_001.nii.gz" in the mask directory
        expected_mask_file = mask_subdir / f"{base_name}.nii.gz"
        
        if not expected_mask_file.exists():
            # Fallback: try to find any .nii.gz file in the directory
            mask_files = list(mask_subdir.glob("*.nii.gz"))
            if not mask_files:
                print(f"Warning: No .nii.gz file found in {mask_subdir}")
                error_count += 1
                continue
            elif len(mask_files) > 1:
                print(f"Warning: Multiple .nii.gz files found in {mask_subdir}, using the first one")
            source_mask = mask_files[0]
        else:
            source_mask = expected_mask_file
        
        # Target mask file to replace
        target_mask = data_subdir / "mask.nii.gz"
        
        if not target_mask.exists():
            print(f"Warning: Target mask.nii.gz not found in {data_subdir}")
            error_count += 1
            continue
            
        try:
            # Create backup of original mask (optional)
            backup_mask = data_subdir / "mask_backup.nii.gz"
            if not backup_mask.exists():
                shutil.copy2(target_mask, backup_mask)
                print(f"Created backup: {backup_mask}")
            
            # Replace the mask file
            shutil.copy2(source_mask, target_mask)
            print(f"Replaced mask for {dir_name}: {source_mask} -> {target_mask}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"Successfully processed: {processed_count} directories")
    print(f"Errors encountered: {error_count} directories")


if __name__ == "__main__":
    from pathlib import Path
    import shutil
    
    # Define the paths
    path_masks = Path(r'D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\cropped-masks\data')
    path_data = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed_crop_n4bc_plhe_fast_full-a\data')
    
    print('Testing fix_mask_location function...')
    print(f'Source path (path_masks): {path_masks}')
    print(f'Target path (path_data): {path_data}')
    print(f'Source path exists: {path_masks.exists()}')
    print(f'Target path exists: {path_data.exists()}')
    print()
    
    # Run the function
    fix_mask_location(path_masks, path_data)    