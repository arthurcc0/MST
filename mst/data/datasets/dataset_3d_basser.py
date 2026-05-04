from pathlib import Path
import pandas as pd
import torch.utils.data as data
import torchio as tio
import torch
import h5py
import numpy as np

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad


class BASSER_Dataset3D(data.Dataset):
    PATH_ROOT = Path(__file__).parent.parent.parent.parent / 'dummy_data' / 'side_v3'
    #PATH_ROOT = Path('/vast/projects/bbruno/breast-imaging/gabriel/vit-gabriel')/'side_v3'
    LABEL = 'Malignant'

    def __init__(
            self,
            path_root=None,
            fold=0,
            split=None,
            fraction=None,
            transform=None,
            image_resize=None,
            resample=None,
            flip=False,
            random_rotate=False,
            image_crop=(224, 224, 32),
            random_center=False,
            noise=False,
            to_tensor=True,
    ):
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)
        self.path_h5 = self.path_root / 'data_basser_compressed.h5'
        self.masks = self.path_root / 'masks_basser_compressed.h5'
        self.split = split

        if transform is None:
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                tio.Flip(1),  # Just for viewing, otherwise upside down
                CropOrPad(image_crop, random_center=random_center, padding_mode='minimum') if image_crop is not None else tio.Lambda(lambda x: x),
                ZNormalization(per_channel=True, per_slice=False, masking_method=lambda x: (x > x.min()) & (x < x.max()), percentiles=(0.5, 99.5)),
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0, 90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0, 1, 2)) if flip else tio.Lambda(lambda x: x),
                tio.Lambda(lambda x: -x if torch.rand((1,),)[0] < 0.5 else x, types_to_apply=[tio.INTENSITY]) if noise else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)
            ])
        else:
            self.transform = transform

        # Get split file
        path_csv = self.path_root / 'splits' / 'split.csv'
        path_or_stream = path_csv
        self.df = self.load_split(path_or_stream, fold=fold, split=split, fraction=fraction)

        # Ensure each patient is loaded only once by dropping duplicates
        self.df = self.df.drop_duplicates(subset=['PatientID'], keep='first').reset_index(drop=True)

        self.item_pointers = self.df.index.tolist()

    def __len__(self):
        return len(self.item_pointers)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        target = item[self.LABEL]
        uid = item['UID']

        # Handle zero-padding for UIDs like "1_left" -> "001_left"
        if '_' in str(uid):
            parts = str(uid).split('_')
            padded_number = parts[0].zfill(3)
            formatted_uid = f"{padded_number}_{'_'.join(parts[1:])}"
        else:
            formatted_uid = str(uid).zfill(3)

        # For HDF5 access, remove laterality suffix since HDF5 keys don't include it
        if '_' in formatted_uid:
            formatted_uid = formatted_uid.split('_')[0]
        else:
            formatted_uid = formatted_uid
        
        patient_id = f'Breast_MRI_{formatted_uid}'
        scan_name = 'sub'

        with h5py.File(self.path_h5, 'r') as f:
            patient_group = f[patient_id]
            data = patient_group[scan_name][()]
            affine = patient_group[f"{scan_name}_affine"][()]
            # Try to read post-contrast volume for final computation
            post = patient_group['post_1'][()] if 'post_1' in patient_group else None
            # Try to read pre-contrast volume for on-the-fly BPE
            pre = patient_group['pre'][()] if 'pre' in patient_group else None

        # Optionally compute final volume using masks if available
        final_img = None
        try:
            if hasattr(self, 'masks') and Path(self.masks).exists():
                with h5py.File(self.masks, 'r') as fm:
                    if patient_id in fm:
                        gm = fm[patient_id]
                        # Resolve mask keys
                        fgt_key = 'fgt' if 'fgt' in gm else ('FGT' if 'FGT' in gm else ('fgt_mask' if 'fgt_mask' in gm else None))
                        bpe_key = 'bpe_mask' if 'bpe_mask' in gm else ('bpe' if 'bpe' in gm else None)
                        breast_key = 'breast' if 'breast' in gm else ('breast_mask' if 'breast_mask' in gm else None)

                        fgt = (gm[fgt_key][()] > 0).astype(np.float32) if fgt_key else None
                        bpe = (gm[bpe_key][()] > 0).astype(np.float32) if bpe_key else None
                        breast = (gm[breast_key][()] > 0).astype(np.float32) if breast_key else None

                        # Base for final computation is post if available; else fall back to subtraction image
                        base = post.astype(np.float32) if post is not None else data.astype(np.float32)
                        final_vol = base
                        if bpe is not None:
                            final_vol = final_vol * bpe
                        if fgt is not None:
                            final_vol = final_vol * fgt
                        if breast is not None:
                            final_vol = final_vol * breast

                        final_img = tio.ScalarImage(tensor=final_vol.astype(np.float32), affine=affine)
        except Exception:
            # Fail silently if masks or keys are unavailable; we still return 'source'
            final_img = None
 
        # Ensure data is in a format torchio understands (e.g., float32)
        img = tio.ScalarImage(tensor=data.astype(np.float32), affine=affine)
 
        img = self.transform(img)
        if final_img is not None:
            final_img = self.transform(final_img)
 
        out = {'uid': uid, 'source': img, 'target': target}
        if final_img is not None:
            out['final'] = final_img
        return out


    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df