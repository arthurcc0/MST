from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch
import nibabel as nib
import numpy as np

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad
def identity_transform(x):
    return x

def znorm_masking_method(x):
    return (x > x.min()) & (x < x.max())

def random_negate_intensity_transform(x):
    return -x if torch.rand((1,),)[0] < 0.5 else x


class DUKE_Dataset3D(data.Dataset):
    PATH_ROOT = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI')
    LABEL = 'Malignant'

    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            image_resize = None,
            resample=None,
            flip = False,
            random_rotate=False,
            image_crop = (224, 224, 32),
            random_center=False,
            noise=False, 
            to_tensor = True,
            get_segmentation=False,
            clinical_notes_path=None,  
            use_clinical_notes=False   
        ):
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)
        self.path_root_data = self.path_root/'preprocessed_crop-a'/'data'
        self.split = split 
        self.get_segmentation = get_segmentation 
        self.use_clinical_notes = use_clinical_notes
        self.clinical_notes_df = None
        
        if use_clinical_notes and clinical_notes_path:
            try:
                self.clinical_notes_df = pd.read_csv(clinical_notes_path)
                print(f"Loaded clinical notes from {clinical_notes_path}")
            except Exception as e:
                print(f"Warning: Could not load clinical notes: {e}")
                self.use_clinical_notes = False

        if transform is None: 
            self.transform = tio.Compose([
                tio.ToCanonical(),
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(identity_transform),
                tio.Resample(resample) if resample is not None else tio.Lambda(identity_transform),
                tio.Flip(1), # Just for viewing, otherwise upside down
                CropOrPad(image_crop, random_center=random_center, padding_mode='minimum') if image_crop is not None else tio.Lambda(identity_transform),
                ZNormalization(per_channel=True, per_slice=False, masking_method=znorm_masking_method, percentiles=(0.5, 99.5)),   # 0.5, 99.5   2.5, 97.5
                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(identity_transform), # WARNING: 1,2 if Subject, 2, 3 if tensor
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(identity_transform),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(identity_transform), # WARNING: Padding mask 
                tio.Lambda(random_negate_intensity_transform, types_to_apply=[tio.INTENSITY]) if noise else tio.Lambda(identity_transform),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(identity_transform),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(identity_transform)             
            ])
        else:
            self.transform = transform


        # Get split file 
        path_csv = self.path_root/'preprocessed_crop-a'/'splits'/'split.csv'
        path_or_stream = path_csv 
        self.df = self.load_split(path_or_stream, fold=fold, split=split, fraction=fraction)
        self.item_pointers = self.df.index.tolist()


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        target = item[self.LABEL]
        uid = item['UID']

        patient_id_unpadded, side = str(uid).split('_', 1)
        patient_id = patient_id_unpadded.zfill(3)

        folder_path = self.path_root_data/f'Breast_MRI_{patient_id}_{side}'
        img_path = folder_path/'sub.nii.gz'

        # Create a TorchIO Subject to handle image and mask together
        # This allows transforms to be applied consistently, and TorchIO handles
        # different interpolation for ScalarImage vs LabelMap automatically.
        if not img_path.exists():
            # Skip this item by trying the next one
            return self.__getitem__((index + 1) % len(self.item_pointers))
    
        subject_dict = {'image': tio.ScalarImage(img_path)}
        if self.get_segmentation:
            mask_path = folder_path/'mask_1.nii.gz'
            if mask_path.exists():
                subject_dict['mask'] = tio.LabelMap(mask_path)

        subject = tio.Subject(**subject_dict)
        
        transformed_subject = self.transform(subject)

        img_tensor = transformed_subject['image'].data

        # Ensure tensor is float and rescale intensity
        img_tensor = img_tensor.float()
        min_val, max_val = img_tensor.min(), img_tensor.max()
        if max_val > min_val:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)

        return_dict = {
            'uid': uid, 
            'source': img_tensor, 
            'target': target, 
            'affine': subject.image.affine
        }

        if 'mask' in transformed_subject:
            mask_tensor = transformed_subject['mask'].data
            return_dict['mask'] = mask_tensor

        if self.use_clinical_notes and self.clinical_notes_df is not None:
            clinical_note = self._get_clinical_note(uid)
            if clinical_note:
                return_dict['text_reports'] = clinical_note
        
        return return_dict


    def _get_clinical_note(self, uid):
        """Get clinical note for a given UID"""
        try:
            # Find matching row in clinical notes DataFrame
            matching_rows = self.clinical_notes_df[self.clinical_notes_df['UID'] == uid]
            
            if not matching_rows.empty:
                clinical_note = matching_rows.iloc[0]['Clinical_Observations']
                return clinical_note
            else:
                # Return a default note if no match found
                return "No specific clinical findings documented for this case."
                
        except Exception as e:
            print(f"Warning: Could not retrieve clinical note for {uid}: {e}")
            return None

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df