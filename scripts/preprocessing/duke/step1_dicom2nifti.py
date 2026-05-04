

from pathlib import Path 
import logging  
import pandas as pd 
from multiprocessing import Pool

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





# Logging 
# path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
# s_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler(path_log_file, 'w')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[s_handler, f_handler])

PLOT_PATH = Path('./plots_duke')

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

        dicom_names = reader.GetGDCMSeriesFileNames(str(path_series_absolute))
        reader.SetFileNames(dicom_names)
        img_nii = reader.Execute()
        dummy_arr = sitk.GetArrayFromImage(img_nii)
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
        img_nii = sitk.GetImageFromArray(dummy_arr)
        
        metadata = dataset2dict(ds)
        # dummy_img = sitk.GetImageFromArray(dummy_arr)
        # arr_2 = sitk.GetArrayFromImage(dummy_img)
        # print("arr_2.min(): {}, arr_2.max(): {}".format(arr_2.min(), arr_2.max()))
        # print("arr_2 type: {}".format(arr_2.dtype))
        # print("------------------------------------------------------")
        patient_id_folder_name = path_series_absolute.parts[-3] 
        path_out_dir = path_root_out_data / patient_id_folder_name
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write NIfTI file
        filename = seq_name 
        path_file = path_out_dir / f'{filename}.npy'
        logger.info(f"Writing file: {path_file}")
        np.save(path_file, dummy_arr)

        metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        return metadata

    except Exception as e:
        logger.warning(f"Error processing series '{seq_name}' in: {path_series_absolute}")
        logger.warning(str(e))






if __name__ == "__main__":
    # Setting 
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI') # Path to the Duke Breast Cancer MRI dataset

    path_root_in = path_root/'manifest-1654812109500'
    path_root_out = path_root/'preprocessed-v1'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
   
    path_root_in_segmentation = path_root/'segmentations'/'manifest-1654811613950'
    path_root_out_segmentation = path_root_out/'data'
    path_root_out_segmentation.mkdir(parents=True, exist_ok=True)
    
    # Init reader 
    # reader = sitk.ImageSeriesReader() # Moved into series2nifti function

    # Note: Contains path to every single dicom file 
    # WARNING: reading this .xlsx file takes some time 
    df_path2name = pd.read_excel(path_root/'Breast-Cancer-MRI-filepath_filename-mapping.xlsx') 
    aux_df = pd.read_excel(path_root/'my-mapping.xlsx')

    # Use aux_df to find segmentation files
    df_path2name['SequenceName'] = aux_df['SequenceName']
    df_path2name['classic_path'] = aux_df['classic_path']

    # Now, we can safely filter by SequenceName
    seg_mask = df_path2name['SequenceName'] == 'Segmentation'
    seg_df = df_path2name[seg_mask].copy()
    other_df = df_path2name[~seg_mask].copy()

    # Process segmentation files
    seg_df['PatientID'] = seg_df['original_path_and_filename'].str.split('/').apply(lambda x: int(x[1].rsplit('_', 1)[1]))
    seg_df['SequenceName'] = 'mask_1' # Rename as requested
    seg_df['classic_path'] = seg_df['classic_path'].apply(lambda p: str(Path(p).parent))

    # Process other files
    seq_paths_split = other_df['original_path_and_filename'].str.split('/')
    other_df['PatientID'] = seq_paths_split.apply(lambda x: int(x[1].rsplit('_', 1)[1]))
    other_df['SequenceName'] = seq_paths_split.apply(lambda x: x[2]) # Derive from path
    other_df['classic_path'] = other_df['classic_path'].apply(lambda p: str(Path(p).parent))

    # Combine the processed dataframes
    final_df = pd.concat([other_df, seg_df], ignore_index=True)

    # Keep only necessary columns and remove duplicates
    final_df = final_df[['PatientID', 'SequenceName', 'classic_path', 'original_path_and_filename']].copy()
    final_df = final_df.drop_duplicates(subset=['PatientID', 'SequenceName'], keep='first')

    # Save the final mapping and create the series list for processing
    final_df.to_csv(path_root_out/'Breast-Cancer-MRI-filepath_filename-mapping.csv', index=False)
    series_df = pd.read_csv(path_root_out/'Breast-Cancer-MRI-filepath_filename-mapping.csv')
    series = list(zip(series_df['SequenceName'], series_df['classic_path']))

    # Validate 
    print("Number Series: ", len(series), "of 5034 (5034+127=5161) ")

    # Split series into segmentations and images
    segmentation_series = [s for s in series if s[0] == 'mask_1']
    image_series = [s for s in series if s[0] != 'mask_1']

    # Prepare processors
    segmentation_processor = functools.partial(series2nifti,
                                       path_root_in_str=str(path_root_in_segmentation),
                                       path_root_out_data_str=str(path_root_out_segmentation))
    
    series_processor = functools.partial(series2nifti,
                                       path_root_in_str=str(path_root_in),
                                       path_root_out_data_str=str(path_root_out_data))

    # Process segmentations
    print(f"Processing {len(segmentation_series)} segmentation series...")
    metadata_segmentation_list = []
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(segmentation_processor, segmentation_series), total=len(segmentation_series)):
            if meta is not None:
                metadata_segmentation_list.append(meta)

    # Process images
    print(f"Processing {len(image_series)} image series...")
    metadata_list = []
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series_processor, image_series), total=len(image_series)):
            if meta is not None:
                metadata_list.append(meta)

    # Combine metadata and save
    all_metadata = metadata_list + metadata_segmentation_list
    df = pd.DataFrame(all_metadata)
    df.to_csv(path_root_out/'metadata.csv', index=False)

    # Check export 
    num_series = len([path for path in path_root_out_data.rglob('*.nii.gz')])
    print("Number Series: ", num_series, "of 5034 (5034+127=5161) ")


