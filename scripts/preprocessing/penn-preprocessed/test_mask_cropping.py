from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import sys
import numpy as np
import nibabel as nib

# Add the parent directory to the path to allow importing from sibling scripts
sys.path.append(str(Path(__file__).parent))

# Import the core processing function from the original script
from crop_masks import process_mask

def test_crop_images_sample():
    """
    Tests the cropping logic on a small sample of images (the first 5).
    """
    # Input directory is the same as the main script
    image_dir = Path(r'D:\PENN-MRI\pennmasked\data')
    
    # Use a separate directory for test outputs
    save_dir = Path(r'D:\PENN-MRI\penn_masked_cropped_TEST\data')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test output will be saved to: {save_dir}")

    # Get all images, but only take the first 5 for the test
    all_images = natsorted(list(image_dir.glob('*.nii.gz')))
    images_to_process = all_images[:5]
    
    if not images_to_process:
        print("No images found in the directory to test. Please check the path.")
        return

    print(f"Found {len(all_images)} total images. Testing with the first {len(images_to_process)}.")

    # Process the sample without multiprocessing for simpler debugging
    for image_path in tqdm(images_to_process, desc="Processing sample images"):
        # Call the function to process the mask, with rotation enabled
        process_mask(image_path, str(save_dir), rotate_img=True)


    print("Finished processing the sample.")

if __name__ == "__main__":
    test_crop_images_sample()
