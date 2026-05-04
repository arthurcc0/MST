import inspect
from pathlib import Path
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad
def identity_transform(x):
    return x

def znorm_masking_method(x):
    return (x > x.min()) & (x < x.max())

def random_negate_intensity_transform(x):
    return -x if torch.rand((1,),)[0] < 0.5 else x


import pytorch_lightning as pl

class PENN_DataModule(pl.LightningDataModule):
    def __init__(self, path_root=None, fold=0, fraction=None, batch_size=32, num_workers=4, only_malignants=False, with_laterality=False, use_clinical_notes=False, clinical_notes_path=None, **kwargs):
        super().__init__()
        self.path_root = path_root
        self.fold = fold
        self.fraction = fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.only_malignants_train = only_malignants
        self.with_laterality = with_laterality
        self.use_clinical_notes = use_clinical_notes
        self.clinical_notes_path = clinical_notes_path

        # Forward only kwargs that PENN_Dataset3D.__init__ actually accepts.
        # Using an allowlist (inspected from the dataset class signature) makes
        # this resilient to new CLI args added in main_train.py.
        _allowed = set(inspect.signature(PENN_Dataset3D.__init__).parameters) - {'self', 'df', 'use_clinical_notes', 'clinical_notes_path'}
        self.dataset_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}

    def setup(self, stage=None):
        # Load full dataset split
        path_csv = PENN_Dataset3D.default_split_csv_path()
        df_full = PENN_Dataset3D.load_split(path_csv, fold=self.fold, fraction=self.fraction)

        # Split data
        df_train = df_full[df_full['Split'] == 'train']
        df_val = df_full[df_full['Split'] == 'val']

        # Split data
        df_train_all = df_full[df_full['Split'] == 'train']
        df_val = df_full[df_full['Split'] == 'val']

        # Filter training data if needed
        if self.only_malignants_train:
            df_train = df_train_all[df_train_all['Malignant'] == 1].copy()
        else:
            df_train = df_train_all

        self.train_dataset = PENN_Dataset3D(df_train, use_clinical_notes=self.use_clinical_notes, clinical_notes_path=self.clinical_notes_path, **self.dataset_kwargs)
        self.val_dataset = PENN_Dataset3D(df_val, use_clinical_notes=self.use_clinical_notes, clinical_notes_path=self.clinical_notes_path, **self.dataset_kwargs)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                               shuffle=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                               shuffle=False, persistent_workers=self.num_workers > 0)


class PENN_Dataset3D(data.Dataset):
    PATH_ROOT = Path(r'\\10.156.155.77\mccarthy_lab\shared\mri_preproc\n4bc')
    LABEL = 'Malignant'
    AUX_PATH = Path(r'D:\Users\arthur\Data\MST_birads4')
    SPLIT_CSV_NAME = 'new_penn_datasplit.csv'

    @classmethod
    def default_split_csv_path(cls):
        return cls.AUX_PATH / cls.SPLIT_CSV_NAME

    def __init__(
            self,
            df, # DataFrame should be passed directly
            path_root=None,
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
            target_col=None,
            filter_col=None,
            use_clinical_notes=False,
            clinical_notes_path=None
        ):
        self.path_root_data = self.AUX_PATH/'final_cropped_and_masked_data'
        self.get_segmentation = get_segmentation
        self.use_clinical_notes = use_clinical_notes
        self.clinical_notes_df = None

        if self.use_clinical_notes and clinical_notes_path:
            try:
                self.clinical_notes_df = pd.read_csv(clinical_notes_path)
                # Assuming 'dummy_acc' in the CSV corresponds to 'UID' and can be treated as integer.
                self.clinical_notes_df['dummy_acc'] = self.clinical_notes_df['dummy_acc'].astype(int)
                self.clinical_notes_df.set_index('dummy_acc', inplace=True)
                print(f"Loaded clinical notes from {clinical_notes_path}")
            except Exception as e:
                print(f"Warning: Could not load clinical notes: {e}")
                self.use_clinical_notes = False

        if target_col is not None:
            self.LABEL = target_col 

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


        self.df = df.copy()

        # Filter out missing files and degenerate (constant-intensity) volumes.
        # The latter happens when step2b's image x mask leaves a side empty,
        # which would crash ZNormalization at training time.
        self.df['img_path'] = self.df['UID'].apply(lambda uid: self.path_root_data / uid / 'sub.nii.gz')
        self.df = self.df[self.df['img_path'].apply(self._is_usable)].reset_index(drop=True)

        self.item_pointers = self.df.index.tolist()

    @staticmethod
    def _is_usable(path: Path) -> bool:
        """File exists and the volume isn't entirely constant (e.g. all zeros)."""
        if not path.exists():
            return False
        try:
            import nibabel as nib
            arr = nib.load(str(path)).get_fdata()
            return bool(arr.size) and float(arr.max()) > float(arr.min())
        except Exception:
            return False


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        target = item[self.LABEL]
        uid = item['UID']

        # Construct the path to the patient's data folder
        folder_path = self.path_root_data / uid
        img_path = folder_path/'sub.nii.gz'

        if not img_path.exists():
            # Fall through to the next sample so the loader doesn't crash on
            # a partially-preprocessed dataset.
            print(f"[PENN_Dataset3D] missing file for UID {uid}; skipping.", flush=True)
            return self.__getitem__((index + 1) % len(self.item_pointers))

        try:
            subject = tio.Subject(image=tio.ScalarImage(img_path))
            transformed_subject = self.transform(subject)
        except RuntimeError as e:
            # Most common cause: ZNormalization complains "Standard deviation
            # is 0" when the masked volume is constant (e.g. all zeros after
            # mask x image). Skip this sample and try the next one.
            print(f"[PENN_Dataset3D] transform failed for UID {uid} ({e}); skipping.", flush=True)
            return self.__getitem__((index + 1) % len(self.item_pointers))

        img_tensor = transformed_subject['image'].data

        return_dict = {'source': img_tensor, 'target': target, 'uid': uid}

        if self.use_clinical_notes and self.clinical_notes_df is not None:
            clinical_note = self._get_clinical_note(uid)
            return_dict['text_reports'] = clinical_note

        return return_dict


    def _get_clinical_note(self, uid):
        """Get clinical note for a given UID"""
        try:
            # UID from the datasplit might be a string, convert to int for lookup
            uid_int = int(uid.split('_')[0])
            if uid_int in self.clinical_notes_df.index:
                clinical_note = self.clinical_notes_df.loc[uid_int]['Narrative']
                return clinical_note if pd.notna(clinical_note) else "No specific clinical findings documented for this case."
            else:
                return "No specific clinical findings documented for this case."
        except Exception as e:
            print(f"Warning: Could not retrieve clinical note for {uid}: {e}")
            return "No specific clinical findings documented for this case."

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df