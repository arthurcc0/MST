"""Helpers for reading a folder of DICOM slices into a 3D numpy volume.

The volume builder is ``get_axial_view``: it loads a list of ``.dcm`` paths,
sorts them in anatomical order, stacks ``pixel_array`` as (H, W, D), and
returns in-plane spacing plus through-plane spacing.
"""
import os
from natsort import natsorted
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def get_dicom_files(directory):
    """Collect ``*.dcm`` paths under ``directory`` (recursive).

    Natural-sorts filenames so ``2.dcm`` comes before ``10.dcm``. That order
    is only a convenience; ``get_axial_view`` re-sorts by DICOM geometry.
    """
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    dicom_files = natsorted(dicom_files)

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

def get_axial_view(dicom_file_list):
    """Read a list of DICOM paths and stack them into one 3D volume.

    File names are ignored for order. Slices are sorted by geometry when
    possible so the last axis runs through the stack (typically inferior →
    superior for axial MRI):

    1. ``ImagePositionPatient`` along the slice-normal axis (from IOP).
    2. Else ``SliceLocation``.
    3. Else ``InstanceNumber``.

    Returns
    -------
    volume : np.ndarray
        Shape (H, W, D) from ``pixel_array``.
    pixel_spacing : tuple
        In-plane (row, col) mm from ``PixelSpacing``.
    slice_thickness : float
        Through-plane mm: ``SpacingBetweenSlices`` if present, else
        ``SliceThickness``.
    z_string : str
        Which of those two tags supplied the through-plane spacing.
    """
    dicom_slices = [pydicom.dcmread(f) for f in dicom_file_list]

    # Prefer patient coordinates over filename / instance order.
    if all(hasattr(d, "ImagePositionPatient") for d in dicom_slices):
        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            # IOP is two 3-vectors: image row direction, then column direction,
            # in patient LPS. Their cross product is the slice-to-slice axis.
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = np.array(iop[:3])
            col_vector = np.array(iop[3:])
            normal = np.cross(row_vector, col_vector)
            # Axial ≈ z, sagittal ≈ x, coronal ≈ y. Sort IPP along that axis.
            sort_axis = int(np.argmax(np.abs(normal)))
            dicom_slices.sort(key=lambda d: float(d.ImagePositionPatient[sort_axis]))
        else:
            dicom_slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    elif all(hasattr(d, "SliceLocation") for d in dicom_slices):
        dicom_slices.sort(key=lambda d: float(d.SliceLocation))
    else:
        dicom_slices.sort(key=lambda d: int(d.InstanceNumber))

    try:
        # Last axis is depth so volume[..., k] is one slice.
        # Depending on applciation, stack on axis 0 or -1.
        volume = np.stack([d.pixel_array for d in dicom_slices], axis=-1)

        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = iop[:3]
            col_vector = iop[3:]
            # If the row axis points opposite +X (patient left), the image
            # is stored left-right flipped relative to a standard axial view.
            if row_vector[0] < 0:
                volume = np.flip(volume, axis=1)
            # If the column axis points opposite +Y (patient posterior),
            # flip superior-inferior in-plane (array axis 0).
            if col_vector[1] < 0:
                volume = np.flip(volume, axis=0)
        
        if hasattr(first_slice, 'PixelSpacing'):
            pixel_spacing = tuple(map(float, first_slice.PixelSpacing))
        # SpacingBetweenSlices is the true gap between slice centers when
        # they overlap or have a gap; SliceThickness is the excited slab.
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