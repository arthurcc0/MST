import os
import shutil
import nibabel as nib
from pathlib import Path
import logging

def setup_logging():
    """Setup logging to track operations"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nifti_integrity_check.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_nifti_integrity(file_path):
    """
    Check if a NIfTI file is valid and not corrupted
    
    Args:
        file_path: Path to the .nii.gz file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Try to load the NIfTI file
        img = nib.load(file_path)
        
        # Try to access the data to ensure it's not corrupted
        data = img.get_fdata()
        
        # Check if the data has reasonable dimensions
        if data.size == 0:
            return False
            
        # Check if header is valid
        header = img.header
        if header is None:
            return False
            
        return True
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return False

def verify_folder_integrity(folder_path, logger):
    """
    Check integrity of post.nii.gz and pre.nii.gz files in a folder
    
    Args:
        folder_path: Path to the folder containing the NIfTI files
        logger: Logger instance
        
    Returns:
        bool: True if all files are valid, False if any file is corrupted
    """
    folder_path = Path(folder_path)
    
    # Expected files
    expected_files = ['post.nii.gz', 'pre.nii.gz']
    
    # Check if both files exist
    for file_name in expected_files:
        file_path = folder_path / file_name
        if not file_path.exists():
            logger.warning(f"Missing file: {file_path}")
            return False
    
    # Check integrity of each file
    for file_name in expected_files:
        file_path = folder_path / file_name
        logger.info(f"Checking integrity of: {file_path}")
        
        if not check_nifti_integrity(file_path):
            logger.error(f"Corrupted file detected: {file_path}")
            return False
    
    logger.info(f"All files in {folder_path.name} are valid")
    return True

def process_data_folder(data_folder_path):
    """
    Process all subfolders in the data directory and delete folders with corrupted files.

    Args:
        data_folder_path: Path to the main data folder.
    """
    logger = setup_logging()
    data_folder = Path(data_folder_path)

    if not data_folder.exists():
        logger.error(f"Data folder does not exist: {data_folder}")
        return

    logger.info(f"Starting integrity check for: {data_folder}")

    deleted_count = 0
    valid_count = 0
    total_count = 0

    # Iterate through all subdirectories
    subfolders = [f for f in data_folder.iterdir() if f.is_dir()]
    total_count = len(subfolders)

    for subfolder in subfolders:
        logger.info(f"\nProcessing folder: {subfolder.name}")

        if not verify_folder_integrity(subfolder, logger):
            logger.warning(f"Folder {subfolder.name} failed integrity check and will be deleted.")
            try:
                shutil.rmtree(subfolder)
                logger.info(f"Deleted folder: {subfolder.name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {subfolder.name}: {str(e)}")
        else:
            valid_count += 1

    # Report results
    logger.info(f"\n{'='*50}")
    logger.info("INTEGRITY CHECK RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Total folders processed: {total_count}")
    logger.info(f"Valid folders: {valid_count}")
    logger.info(f"Deleted folders: {deleted_count}")

    if deleted_count > 0:
        logger.info(f"\nSuccessfully deleted {deleted_count} folders.")
    else:
        logger.info("\nNo corrupted folders found. All folders are valid.")

def main():
    """Main function to run the integrity checker"""
    print("NIfTI File Integrity Checker")
    print("="*40)

    # Define the data folder path
    data_folder = Path(r'\\rad-maid-004\D\PENN-MRI\penn-preprocessed2\data')

    try:
        process_data_folder(data_folder)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()