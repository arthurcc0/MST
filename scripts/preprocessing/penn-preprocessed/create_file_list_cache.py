import os
from pathlib import Path
from tqdm import tqdm

def create_file_list_cache():
    """
    Scans the network directory for all .npy files and saves the list to a local cache file.
    This is a slow, one-time operation.
    """
    images_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI\output\preproc\n4bc_plhe')
    cache_file = Path(__file__).parent / 'image_file_cache.txt'

    print(f"Starting to scan network directory: {images_dir}")
    print("This may take a very long time, but you only need to run it once.")

    all_image_paths = []
    # Use os.walk, which is generally more efficient for deep directory structures.
    for root, _, files in tqdm(os.walk(images_dir), desc="Scanning directories"):
        for name in files:
            if name.endswith('.npy'):
                all_image_paths.append(str(Path(root) / name))

    print(f"Found {len(all_image_paths)} total image files.")

    # Save the list to the cache file
    with open(cache_file, 'w') as f:
        for path in all_image_paths:
            f.write(f"{path}\n")

    print(f"Successfully created file list cache at: {cache_file}")

if __name__ == "__main__":
    create_file_list_cache()
