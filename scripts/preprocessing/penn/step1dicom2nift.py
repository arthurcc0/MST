

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root)) 
import logging  
import pandas as pd 
from multiprocessing import Pool
import csv

import numpy as np 
import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk
import functools 
import matplotlib.pyplot as plt 


from common import get_axial_view

# Logging 
# path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
# s_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler(path_log_file, 'w')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[s_handler, f_handler])

PLOT_PATH = Path('./plots_duke')
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
        volume = np.stack([d.pixel_array for d in dicom_slices], axis=0)

        # Optional: Correct image orientation if necessary
        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = np.array(iop[:3])
            col_vector = np.array(iop[3:])
            normal = np.cross(row_vector, col_vector)
            orientation = np.argmax(np.abs(normal))

            # Reorient to axial view if necessary
            if orientation == 0:  # Sagittal
                volume = np.transpose(volume, (0, 2, 1))
            elif orientation == 1:  # Coronal
                pass # Already in (Slices, H, W) which is what we want for Coronal -> Axial

            # Flip based on patient orientation.
            if row_vector[0] < 0:
                volume = np.flip(volume, axis=1) # Flip along X
            if col_vector[1] < 0:
                volume = np.flip(volume, axis=0) # Flip along Y
        
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
        
def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def series2nifti(series_info, path_root_in_str, path_root_out_data_str):
    path_root_in = Path(path_root_in_str)
    path_root_out_data = Path(path_root_out_data_str)
    reader = sitk.ImageSeriesReader() 

    seq_name, path_series_relative = series_info
    path_series_absolute = path_root_in / Path(path_series_relative)

    if not path_series_absolute.is_dir():
        logger.warning(f"Expected directory but found file: {path_series_absolute}")
        return None
    
    try:
        # Read DICOM
        dicom_files = list(path_series_absolute.glob('*.dcm'))
        volume, pixel_spacing, slice_thickness, z_string = get_axial_view(dicom_files)
        print(f" {seq_name}: {volume.shape}, {pixel_spacing}, {slice_thickness}, {z_string}")
        # Read DICOM series to get a template image with correct metadata
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series_absolute))
        reader.SetFileNames(dicom_names)

        # Use the volume from get_axial_view as the pixel data
        # if dummy_arr.min() < 0:
        #     dummy_arr = dummy_arr - dummy_arr.min()
        #     dummy_arr = (dummy_arr/dummy_arr.max()) * 65535
        #     img_nii = sitk.GetImageFromArray(dummy_arr)
        #     img_nii = sitk.CopyInformation(img_nii, reader.GetOutput())
        # img_nii = sitk.Cast(img_nii, sitk.sitkInt16)

        # Read Metadata from the first DICOM file found

        ds = pydicom.dcmread(next(path_series_absolute.glob('*.dcm'), None), stop_before_pixels=True)
        # print("------------------------------------------------------")
        # print("dummy_arr.min(): {}, dummy_arr.max(): {}, ds['BitsStored'].value: {}".format(dummy_arr.min(), dummy_arr.max(), ds['BitsStored'].value))
        
        # if ds['BitsStored'].value == 12: 
        #     if dummy_arr.min() < 0:
        #         dummy_arr = dummy_arr - dummy_arr.min()
        #         dummy_arr = (dummy_arr/dummy_arr.max()) * 4096
        # elif ds['BitsStored'].value == 16:
        #     if dummy_arr.min() < 0:
        #         dummy_arr = dummy_arr - dummy_arr.min()
        #         dummy_arr = (dummy_arr/dummy_arr.max()) * 65535
        # dummy_arr = dummy_arr.astype(np.uint16)
        # # print("dummy_arr.min(): {}, dummy_arr.max(): {}, ds['BitsStored'].value: {}".format(dummy_arr.min(), dummy_arr.max(), ds['BitsStored'].value))
        # print("arr type: {}".format(dummy_arr.dtype))
        # plt.imshow(dummy_arr[56, :, :])
        # plt.savefig(PLOT_PATH/'test_{}_{}.png'.format(seq_name, path_series_absolute.parts[-3]))
        img_nii = sitk.GetImageFromArray(volume)
        
        metadata = dataset2dict(ds)
        # dummy_img = sitk.GetImageFromArray(dummy_arr)
        # arr_2 = sitk.GetArrayFromImage(dummy_img)
        # print("arr_2.min(): {}, arr_2.max(): {}".format(arr_2.min(), arr_2.max()))
        # print("arr_2 type: {}".format(arr_2.dtype))
        # print("------------------------------------------------------")
        # Determine output directory and filename from the sequence name
        patient_id, contrast = seq_name.rsplit('_', 1)
        path_out_dir = path_root_out_data / patient_id
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write NIfTI file
        filename = f"{contrast.lower()}.nii.gz"
        path_file = path_out_dir / filename
        logger.info(f"Writing file: {path_file}")
        sitk.WriteImage(img_nii, path_file)

        metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        return metadata

    except Exception as e:
        logger.warning(f"Error processing series '{seq_name}' in: {path_series_absolute}")
        logger.warning(str(e))






if __name__ == "__main__":
    # Setting 
    path_root = Path(r'\\rad-maid-004\D\PENN-MRI') 
    data_root_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI')

    path_root_in = data_root_dir/'data'
    path_root_out = path_root/'preprocessed'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
   
    # path_root_in_segmentation = path_root/'segmentations'/'manifest-1654811613950'
    # path_root_out_segmentation = path_root_out/'data'
    # path_root_out_segmentation.mkdir(parents=True, exist_ok=True)
    
    # Init reader 
    # reader = sitk.ImageSeriesReader() # Moved into series2nifti function

    # Note: Contains path to every single dicom file 
    # WARNING: reading this .xlsx file takes some time 
    df_mapping = pd.read_csv('penn_mapping.csv', dtype={'PatientID': str})

    # A series is a unique combination of PatientID and Contrast
    df_series = df_mapping[['PatientID', 'Contrast']].drop_duplicates().copy()

    # Define the name for each series (used for the output NIfTI file)
    df_series['SequenceName'] = df_series['PatientID'] + '_' + df_series['Contrast']

    # Define the relative path to the directory containing the DICOM files for the series
    df_series['series_path'] = df_series.apply(lambda row: Path(row['PatientID']) / row['Contrast'], axis=1)

    # Save the processed mapping for reference
    processed_mapping_path = path_root_out / 'penn-dicom-series-mapping.csv'
    df_series.to_csv(processed_mapping_path, index=False)
    logger.info(f"Processed series mapping saved to {processed_mapping_path}")

    # Create the list of series to be processed.
    # Each item is a tuple of (SequenceName, relative_path_to_series_directory)
    image_series = list(zip(df_series['SequenceName'], df_series['series_path'].apply(str)))

    # Validate 
    print("Number of series to process: ", len(image_series))


    # Prepare processors
    # # Prepare processors
    # segmentation_processor = functools.partial(series2nifti,
    #                                    path_root_in_str=str(path_root_in_segmentation),
    #                                    path_root_out_data_str=str(path_root_out_segmentation))
    
    series_processor = functools.partial(series2nifti,
                                       path_root_in_str=str(path_root_in),
                                       path_root_out_data_str=str(path_root_out_data))

    # # Process segmentations
    # print(f"Processing {len(segmentation_series)} segmentation series...")
    # metadata_segmentation_list = []
    # with Pool() as pool:
    #     for meta in tqdm(pool.imap_unordered(segmentation_processor, segmentation_series), total=len(segmentation_series)):
    #         if meta is not None:
    #             metadata_segmentation_list.append(meta)

    # Process images
    print(f"Processing {len(image_series)} image series...")
    metadata_list = []
    c = 0
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series_processor, image_series), total=len(image_series)):
            c += 1
            if c == 1200:
                break
            if meta is not None:
                metadata_list.append(meta)

    # Save metadata
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        output_path = path_root_out / 'metadata.csv'
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Metadata saved to {output_path}")
    else:
        logger.info("No metadata was generated.")

    # Check export 
    num_series_processed = len(list(path_root_out_data.rglob('*.nii.gz')))
    print(f"Conversion complete. {num_series_processed}/{len(image_series)} series were successfully converted.")

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root)) 
import logging  
import pandas as pd 
from multiprocessing import Pool
import csv

import numpy as np 
import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk
import functools 
import matplotlib.pyplot as plt 


from common import get_axial_view

# Logging 
# path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
# s_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler(path_log_file, 'w')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[s_handler, f_handler])

PLOT_PATH = Path('./plots_duke')
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
        volume = np.stack([d.pixel_array for d in dicom_slices], axis=0)

        # Optional: Correct image orientation if necessary
        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = np.array(iop[:3])
            col_vector = np.array(iop[3:])
            normal = np.cross(row_vector, col_vector)
            orientation = np.argmax(np.abs(normal))

            # Reorient to axial view if necessary
            if orientation == 0:  # Sagittal
                volume = np.transpose(volume, (0, 2, 1))
            elif orientation == 1:  # Coronal
                pass # Already in (Slices, H, W) which is what we want for Coronal -> Axial

            # Flip based on patient orientation.
            if row_vector[0] < 0:
                volume = np.flip(volume, axis=1) # Flip along X
            if col_vector[1] < 0:
                volume = np.flip(volume, axis=0) # Flip along Y
        
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
        
def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def series2nifti(series_info, path_root_in_str, path_root_out_data_str):
    path_root_in = Path(path_root_in_str)
    path_root_out_data = Path(path_root_out_data_str)
    reader = sitk.ImageSeriesReader() 

    seq_name, path_series_relative = series_info
    path_series_absolute = path_root_in / Path(path_series_relative)

    if not path_series_absolute.is_dir():
        logger.warning(f"Expected directory but found file: {path_series_absolute}")
        return None
    
    try:
        # Read DICOM
        dicom_files = list(path_series_absolute.glob('*.dcm'))
        volume, pixel_spacing, slice_thickness, z_string = get_axial_view(dicom_files)

        # Skip volumes with x-dimension (width) less than 100
        if volume.shape[2] < 100:
            logger.info(f"Skipping {seq_name} due to small x-dimension: {volume.shape[2]}")
            return None

        print(f" {seq_name}: {volume.shape}, {pixel_spacing}, {slice_thickness}, {z_string}")
        # Read DICOM series to get a template image with correct metadata
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series_absolute))
        reader.SetFileNames(dicom_names)

        # Use the volume from get_axial_view as the pixel data
        # if dummy_arr.min() < 0:
        #     dummy_arr = dummy_arr - dummy_arr.min()
        #     dummy_arr = (dummy_arr/dummy_arr.max()) * 65535
        #     img_nii = sitk.GetImageFromArray(dummy_arr)
        #     img_nii = sitk.CopyInformation(img_nii, reader.GetOutput())
        # img_nii = sitk.Cast(img_nii, sitk.sitkInt16)

        # Read Metadata from the first DICOM file found

        ds = pydicom.dcmread(next(path_series_absolute.glob('*.dcm'), None), stop_before_pixels=True)
        # print("------------------------------------------------------")
        # print("dummy_arr.min(): {}, dummy_arr.max(): {}, ds['BitsStored'].value: {}".format(dummy_arr.min(), dummy_arr.max(), ds['BitsStored'].value))
        
        # if ds['BitsStored'].value == 12: 
        #     if dummy_arr.min() < 0:
        #         dummy_arr = dummy_arr - dummy_arr.min()
        #         dummy_arr = (dummy_arr/dummy_arr.max()) * 4096
        # elif ds['BitsStored'].value == 16:
        #     if dummy_arr.min() < 0:
        #         dummy_arr = dummy_arr - dummy_arr.min()
        #         dummy_arr = (dummy_arr/dummy_arr.max()) * 65535
        # dummy_arr = dummy_arr.astype(np.uint16)
        # # print("dummy_arr.min(): {}, dummy_arr.max(): {}, ds['BitsStored'].value: {}".format(dummy_arr.min(), dummy_arr.max(), ds['BitsStored'].value))
        # print("arr type: {}".format(dummy_arr.dtype))
        # plt.imshow(dummy_arr[56, :, :])
        # plt.savefig(PLOT_PATH/'test_{}_{}.png'.format(seq_name, path_series_absolute.parts[-3]))
        img_nii = sitk.GetImageFromArray(volume)
        
        metadata = dataset2dict(ds)
        # dummy_img = sitk.GetImageFromArray(dummy_arr)
        # arr_2 = sitk.GetArrayFromImage(dummy_img)
        # print("arr_2.min(): {}, arr_2.max(): {}".format(arr_2.min(), arr_2.max()))
        # print("arr_2 type: {}".format(arr_2.dtype))
        # print("------------------------------------------------------")
        # Determine output directory and filename from the sequence name
        patient_id, contrast = seq_name.rsplit('_', 1)
        path_out_dir = path_root_out_data / patient_id
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write NIfTI file
        filename = f"{contrast.lower()}.nii.gz"
        path_file = path_out_dir / filename
        logger.info(f"Writing file: {path_file}")
        sitk.WriteImage(img_nii, path_file)

        metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        return metadata

    except Exception as e:
        logger.warning(f"Error processing series '{seq_name}' in: {path_series_absolute}")
        logger.warning(str(e))






if __name__ == "__main__":
    # Define the base directory for the Penn dataset
    base_dir = Path(r'\\rad-maid-004\D\PENN-MRI')     
    # Define input and output paths relative to the base directory
    path_root_in = base_dir / 'data'
    path_root_out = base_dir / 'preprocessed'
    path_root_out_data = path_root_out / 'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)

    # Load the mapping file from the base directory
    mapping_file = base_dir / 'penn_mapping.csv'
    if not mapping_file.is_file():
        raise FileNotFoundError(f"Mapping file not found at {mapping_file}. Please run penn_mapping.py first.")
    df_mapping = pd.read_csv(mapping_file, dtype={'PatientID': str})

    # A series is a unique combination of PatientID and Contrast
    df_series = df_mapping[['PatientID', 'Contrast']].drop_duplicates().copy()

    # Define the name for each series (used for the output NIfTI file)
    df_series['SequenceName'] = df_series['PatientID'] + '_' + df_series['Contrast']

    # Define the relative path to the directory containing the DICOM files for the series
    df_series['series_path'] = df_series.apply(lambda row: Path(row['PatientID']) / row['Contrast'], axis=1)

    # Save the processed mapping for reference
    processed_mapping_path = path_root_out / 'penn-dicom-series-mapping.csv'
    df_series.to_csv(processed_mapping_path, index=False)
    logger.info(f"Processed series mapping saved to {processed_mapping_path}")

    # Create the list of series to be processed.
    # Each item is a tuple of (SequenceName, relative_path_to_series_directory)
    image_series = list(zip(df_series['SequenceName'], df_series['series_path'].apply(str)))

    # Validate
    print("Number of series to process: ", len(image_series))

    # Prepare processor
    series_processor = functools.partial(series2nifti,
                                       path_root_in_str=str(path_root_in),
                                       path_root_out_data_str=str(path_root_out_data))

    # Process images
    print(f"Processing {len(image_series)} image series...")
    metadata_list = []
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series_processor, image_series), total=len(image_series)):
            if meta is not None:
                metadata_list.append(meta)
                if len(metadata_list) == 2500:
                    break

    # Save metadata
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        output_path = path_root_out / 'metadata.csv'
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Metadata saved to {output_path}")
    else:
        logger.info("No metadata was generated.")

    # Check export
    num_series_processed = len(list(path_root_out_data.rglob('*.nii.gz')))
    print(f"Conversion complete. {num_series_processed}/{len(image_series)} series were successfully converted.")





import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root)) 
import logging  
import pandas as pd 
from multiprocessing import Pool
import csv

import numpy as np 
import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk
import functools 
import matplotlib.pyplot as plt 


from common import get_axial_view

# Logging 
# path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
# s_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler(path_log_file, 'w')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[s_handler, f_handler])

PLOT_PATH = Path('./plots_duke')
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
        volume = np.stack([d.pixel_array for d in dicom_slices], axis=0)

        # Optional: Correct image orientation if necessary
        first_slice = dicom_slices[0]
        if hasattr(first_slice, "ImageOrientationPatient"):
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_vector = np.array(iop[:3])
            col_vector = np.array(iop[3:])
            normal = np.cross(row_vector, col_vector)
            orientation = np.argmax(np.abs(normal))

            # Reorient to axial view if necessary
            if orientation == 0:  # Sagittal
                volume = np.transpose(volume, (0, 2, 1))
            elif orientation == 1:  # Coronal
                pass # Already in (Slices, H, W) which is what we want for Coronal -> Axial

            # Flip based on patient orientation.
            if row_vector[0] < 0:
                volume = np.flip(volume, axis=1) # Flip along X
            if col_vector[1] < 0:
                volume = np.flip(volume, axis=0) # Flip along Y
        
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
        
def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def series2nifti(series_info, path_root_in_str, path_root_out_data_str):
    path_root_in = Path(path_root_in_str)
    path_root_out_data = Path(path_root_out_data_str)
    reader = sitk.ImageSeriesReader() 

    seq_name, path_series_relative = series_info
    path_series_absolute = path_root_in / Path(path_series_relative)

    if not path_series_absolute.is_dir():
        logger.warning(f"Expected directory but found file: {path_series_absolute}")
        return None
    
    try:
        # Read DICOM
        dicom_files = list(path_series_absolute.glob('*.dcm'))
        volume, pixel_spacing, slice_thickness, z_string = get_axial_view(dicom_files)
        print(f" {seq_name}: {volume.shape}, {pixel_spacing}, {slice_thickness}, {z_string}")


        # Read DICOM series to get a template image with correct metadata
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series_absolute))
        reader.SetFileNames(dicom_names)

        # Use the volume from get_axial_view as the pixel data
        # if dummy_arr.min() < 0:
        #     dummy_arr = dummy_arr - dummy_arr.min()
        #     dummy_arr = (dummy_arr/dummy_arr.max()) * 65535
        #     img_nii = sitk.GetImageFromArray(dummy_arr)
        #     img_nii = sitk.CopyInformation(img_nii, reader.GetOutput())
        # img_nii = sitk.Cast(img_nii, sitk.sitkInt16)

        # Read Metadata from the first DICOM file found

        ds = pydicom.dcmread(next(path_series_absolute.glob('*.dcm'), None), stop_before_pixels=True)
        # print("------------------------------------------------------")
        # print("dummy_arr.min(): {}, dummy_arr.max(): {}, ds['BitsStored'].value: {}".format(dummy_arr.min(), dummy_arr.max(), ds['BitsStored'].value))
        
        # if ds['BitsStored'].value == 12: 
        #     if dummy_arr.min() < 0:
        #         dummy_arr = dummy_arr - dummy_arr.min()
        #         dummy_arr = (dummy_arr/dummy_arr.max()) * 4096
        # elif ds['BitsStored'].value == 16:
        #     if dummy_arr.min() < 0:
        #         dummy_arr = dummy_arr - dummy_arr.min()
        #         dummy_arr = (dummy_arr/dummy_arr.max()) * 65535
        # dummy_arr = dummy_arr.astype(np.uint16)
        # # print("dummy_arr.min(): {}, dummy_arr.max(): {}, ds['BitsStored'].value: {}".format(dummy_arr.min(), dummy_arr.max(), ds['BitsStored'].value))
        # print("arr type: {}".format(dummy_arr.dtype))
        # plt.imshow(dummy_arr[56, :, :])
        # plt.savefig(PLOT_PATH/'test_{}_{}.png'.format(seq_name, path_series_absolute.parts[-3]))
        img_nii = sitk.GetImageFromArray(volume)
        
        metadata = dataset2dict(ds)
        # dummy_img = sitk.GetImageFromArray(dummy_arr)
        # arr_2 = sitk.GetArrayFromImage(dummy_img)
        # print("arr_2.min(): {}, arr_2.max(): {}".format(arr_2.min(), arr_2.max()))
        # print("arr_2 type: {}".format(arr_2.dtype))
        # print("------------------------------------------------------")
        # Determine output directory and filename from the sequence name
        patient_id, contrast = seq_name.rsplit('_', 1)
        path_out_dir = path_root_out_data / patient_id
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write NIfTI file
        filename = f"{contrast.lower()}.nii.gz"
        path_file = path_out_dir / filename
        logger.info(f"Writing file: {path_file}")
        sitk.WriteImage(img_nii, path_file)

        metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        return metadata

    except Exception as e:
        logger.warning(f"Error processing series '{seq_name}' in: {path_series_absolute}")
        logger.warning(str(e))






if __name__ == "__main__":
    # Setting 
    path_root = Path(r'\\rad-maid-004\D\PENN-MRI') 
    data_root_dir = Path(r'\\10.156.155.77\mccarthy_lab\MRI')

    path_root_in = data_root_dir/'data'
    path_root_out = path_root/'preprocessed'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
   
    # path_root_in_segmentation = path_root/'segmentations'/'manifest-1654811613950'
    # path_root_out_segmentation = path_root_out/'data'
    # path_root_out_segmentation.mkdir(parents=True, exist_ok=True)
    
    # Init reader 
    # reader = sitk.ImageSeriesReader() # Moved into series2nifti function

    # Note: Contains path to every single dicom file 
    # WARNING: reading this .xlsx file takes some time 
    df_mapping = pd.read_csv('penn_mapping.csv', dtype={'PatientID': str})

    # Load patient info and filter by laterality
    df_patient_info = pd.read_csv('mapped_patient_info.csv', dtype={'dummy_acc': str})
    df_patient_info = df_patient_info[df_patient_info['laterality'].isin([1.0, 2.0])]
    valid_patients = df_patient_info['dummy_acc'].unique()

    # Filter the mapping dataframe to only include valid patients
    df_mapping = df_mapping[df_mapping['PatientID'].isin(valid_patients)]

    # A series is a unique combination of PatientID and Contrast
    df_series = df_mapping[['PatientID', 'Contrast']].drop_duplicates().copy()

    # Define the name for each series (used for the output NIfTI file)
    df_series['SequenceName'] = df_series['PatientID'] + '_' + df_series['Contrast']

    # Define the relative path to the directory containing the DICOM files for the series
    df_series['series_path'] = df_series.apply(lambda row: Path(row['PatientID']) / row['Contrast'], axis=1)

    # Save the processed mapping for reference
    processed_mapping_path = path_root_out / 'penn-dicom-series-mapping.csv'
    df_series.to_csv(processed_mapping_path, index=False)
    logger.info(f"Processed series mapping saved to {processed_mapping_path}")

    # Create the list of series to be processed.
    # Each item is a tuple of (SequenceName, relative_path_to_series_directory)
    image_series = list(zip(df_series['SequenceName'], df_series['series_path'].apply(str)))

    # Validate 
    print("Number of series to process: ", len(image_series))


    # Prepare processors
    # # Prepare processors
    # segmentation_processor = functools.partial(series2nifti,
    #                                    path_root_in_str=str(path_root_in_segmentation),
    #                                    path_root_out_data_str=str(path_root_out_segmentation))
    
    series_processor = functools.partial(series2nifti,
                                       path_root_in_str=str(path_root_in),
                                       path_root_out_data_str=str(path_root_out_data))

    # # Process segmentations
    # print(f"Processing {len(segmentation_series)} segmentation series...")
    # metadata_segmentation_list = []
    # with Pool() as pool:
    #     for meta in tqdm(pool.imap_unordered(segmentation_processor, segmentation_series), total=len(segmentation_series)):
    #         if meta is not None:
    #             metadata_segmentation_list.append(meta)

    # Process images
    print(f"Processing {len(image_series)} image series...")
    metadata_list = []
    c = 0
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series_processor, image_series), total=len(image_series)):
            c += 1
            if c == 2500:
                break
            if meta is not None:
                metadata_list.append(meta)

    # Save metadata
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        output_path = path_root_out / 'metadata.csv'
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        logger.info(f"Metadata saved to {output_path}")
    else:
        logger.info("No metadata was generated.")

    # Check export 
    num_series_processed = len(list(path_root_out_data.rglob('*.nii.gz')))
    print(f"Conversion complete. {num_series_processed}/{len(image_series)} series were successfully converted.")


