import os
from natsort import natsorted
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def get_dicom_files(directory):
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    dicom_files = natsorted(dicom_files) # Sort files in natural order

    return dicom_files

# # Authors original code (it does not work if file list is not sorted)
# def get_axial_view(dicom_file_list):
    
#     # Saving the values of first two image positions
#     # This is used to orient inferior to superior
#     first_image_position = 0
#     second_image_position = 0

#     dicom_data_list = []
#     for i in range(len(dicom_file_list)):
#         dicom_data = pydicom.dcmread(dicom_file_list[i])
        
#         if i == 0:
#             first_image_position = dicom_data[0x20, 0x32].value[-1]
#         elif i == 1:
#             second_image_position = dicom_data[0x20, 0x32].value[-1]

#         dicom_data_list.append(dicom_data.pixel_array)
        
#     # Stack in numpy array
#     image_array = np.stack(dicom_data_list, axis=-1)

#     if round(dicom_data[0x20, 0x37].value[0], 0) == -1:
#         image_array = np.rot90(image_array, 2)

#     return image_array
def bread_first_search(elements, visited_set: set):
    queue = []
    def bfs(element):
        if element in visited_set:
            return
        visited_set.add(element)
        queue.append(element)
        for neighbor in elements[element]:
            bfs(neighbor)
    for element in elements:
        if element not in visited_set:
            bfs(element)

def depth_first_search_interface(elements, visited_set: set):
    def dfs(element):
        if element in visited_set:
            return
        visited_set.add(element)
        for neighbor in elements[element]:
            dfs(neighbor)

    for element in elements:
        if element not in visited_set:
            dfs(element)

# # Identify and rotate sagittal views
# if abs(iop[0]) < 0.1 and abs(iop[4]) < 0.1:  # Sagittal detection 
def get_axial_view(dicom_file_list):
    dicom_slices = [pydicom.dcmread(f) for f in dicom_file_list]

    # Sorting logic based on available metadata
    if all(hasattr(d, "ImagePositionPatient") for d in dicom_slices):
        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            # Extract orientation vectors
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = np.array(iop[:3])
            col_vector = np.array(iop[3:])
            # Compute the normal vector (slice direction)
            normal = np.cross(row_vector, col_vector)
            # Determine which coordinate (x=0, y=1, or z=2) varies the most
            sort_axis = int(np.argmax(np.abs(normal)))
            dicom_slices.sort(key=lambda d: float(d.ImagePositionPatient[sort_axis]))
        else:
            # Fallback to z-axis sorting if orientation is not available
            dicom_slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    elif all(hasattr(d, "SliceLocation") for d in dicom_slices):
        dicom_slices.sort(key=lambda d: float(d.SliceLocation))
    else:
        dicom_slices.sort(key=lambda d: int(d.InstanceNumber))

    # Stack the slices into a 3D numpy array
    try:
        volume = np.stack([d.pixel_array for d in dicom_slices], axis=-1)

        # Optional: Correct image orientation if necessary
        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = iop[:3]
            col_vector = iop[3:]
            
            # Check if the series is sagittal: normal dominated by x (index 0)
            # if np.abs(normal[0]) > np.abs(normal[1]) and np.abs(normal[0]) > np.abs(normal[2]):
            #     volume = np.transpose(volume, (2, 1, 0))
            #     volume = np.rot90(np.fliplr(volume), k=1)
            # else:
            # # For axial (or other) views, you might need to flip the image based on orientation.
            if row_vector[0] < 0:
                volume = np.flip(volume, axis=1)
            if col_vector[1] < 0:
                volume = np.flip(volume, axis=0)
        
        if hasattr(first_slice, 'PixelSpacing'):
            pixel_spacing = tuple(map(float, first_slice.PixelSpacing))
        if hasattr(first_slice, 'SpacingBetweenSlices'):
            slice_thickness = float(first_slice.SpacingBetweenSlices)
            z_string = 'Using SpacingBetweenSlices'
        else:
            slice_thickness = float(first_slice.SliceThickness)
            z_string = 'Using SliceThickness'

        return volume, pixel_spacing, slice_thickness, z_string
    except Exception as e:
        print(f"Error reading dicom files: {e}")

def save_image(image, filename):
    """
    Saves a 2D image (numpy array) as a PNG using matplotlib.
    """
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_slices_check(volume):
    depth = volume.shape[2]
    slices = np.array([
        volume[:, :, int(depth * 0.75)],
        volume[:, :, depth // 2],
        volume[:, :, int(depth * 0.25)]
    ])
    return slices

def plot_grid(array_input, array_output, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    
    axes[0, 0].imshow(array_input[0], cmap='gray')
    axes[0, 0].set_title("input (0.75)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(array_input[1], cmap='gray')
    axes[0, 1].set_title("input (0.5)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(array_input[2], cmap='gray')
    axes[0, 2].set_title("input (0.25)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(array_output[0], cmap='gray')
    axes[1, 0].set_title("output (0.75)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(array_output[1], cmap='gray')
    axes[1, 1].set_title("output (0.5)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(array_output[2], cmap='gray')
    axes[1, 2].set_title("output (0.25)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
    """
    Normalize the intensity of an image array by cutting off min and max values 
    to a certain percentile and set all values above/below that percentile to 
    the new max/min. 

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    min_cutoff: float
        Minimum percentile of image to keep. (0.1% = 0.001)
    max_cutoff: float
        Maximum percentile of image to keep. (0.1% = 0.001)

    Returns
    -------
    np.array
        Normalized image

    """

    # Sort image values
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * min_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def zscore_image(image_array):
    """
    Convert intensity values in an image to zscores:
    zscore = (intensity_value - mean) / standard_deviation

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    Returns
    -------
    np.array
        Image with zscores for values

    """

    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    return image_array