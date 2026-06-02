import torchio as tio 
from typing import Iterable, Tuple, Union, List, Optional, Sequence, Dict
from numbers import Number
import nibabel as nib 
import numpy as np
from torchio.types import TypeRangeFloat, TypeTripletInt
from torchio.transforms.transform import TypeMaskingMethod 
from torchio import Subject, Image
import torch 



class SubjectToTensor(object):
    """Transforms TorchIO Subjects into a Python dict and changes axes order from TorchIO to Torch"""
    def __call__(self, subject: Subject):
        return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in subject.items()}

class ImageToTensor(object):
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1,-1)

class ImageOrSubjectToTensor(object):
    """Depending on the input, it will either run SubjectToTensor or ImageToTensor"""
    def __call__(self, input: Union[Image, Subject]):
        if isinstance(input, Subject):
            return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in input.items()}
        else:
            return input.data.swapaxes(1,-1)

def parse_per_channel(per_channel, channels):
    if isinstance(per_channel, bool):
        if per_channel == True:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))] 
    else:
        return per_channel 

class ZNormalization(tio.ZNormalization):
    """Add option 'per_channel' to apply znorm for each channel independently and percentiles to clip values first"""
    def __init__(
        self,
        percentiles: TypeRangeFloat = (0, 100),
        per_channel=True,
        per_slice=False,
        masking_method: TypeMaskingMethod = None,
        **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel
        self.per_slice =  per_slice


    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        per_slice = parse_per_channel(self.per_slice, image.shape[-1])

        image.set_data(
            torch.cat([
                torch.cat([
                    self._znorm(image.data[chs,][:,:,:, sl,], mask[chs,][:,:,:, sl,], image_name, image.path)
                for sl in per_slice], dim=-1)
            for chs in per_channel ])
        )
  

    def _znorm(self, image_data, mask, image_name, image_path):
        if not mask.any():
            return torch.zeros_like(image_data)
        cutoff = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles)/100.0)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)

        standardized = self.znorm(image_data, mask)
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image_path})'
            )
            raise RuntimeError(message)
        return standardized



class RescaleIntensity(tio.RescaleIntensity):
    """Add option 'per_channel' to apply rescale for each channel independently"""
    def __init__(
        self,
        out_min_max: TypeRangeFloat = (0, 1),
        percentiles: TypeRangeFloat = (0, 100),
        masking_method: TypeMaskingMethod = None,
        in_min_max: Optional[Tuple[float, float]] = None,
        per_channel=True, # Bool or List of tuples containing channel indices that should be normalized together 
        per_slice=False,
        **kwargs
    ):
        super().__init__(out_min_max, percentiles, masking_method, in_min_max, **kwargs)
        self.per_channel=per_channel
        self.per_slice = per_slice
        self.in_min_max_backup = in_min_max # Fix Bug: self.in_min_max is overwritten by function call -> reset to initial value 

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
    ) -> None:
        self.in_min_max = self.in_min_max_backup # FIX BUG 

        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        per_slice = parse_per_channel(self.per_slice, image.shape[-1])
        
        image.set_data(
            torch.cat([
                torch.cat([
                    self.rescale(image.data[chs,][:,:,:, sl,], mask[chs,][:,:,:, sl,], image_name)
                for sl in per_slice], dim=-1)
            for chs in per_channel ])
        )


class EnsureShapeMultiple(tio.EnsureShapeMultiple):
    """Add option 'padding_mode' """
    def __init__(self, target_multiple, *, method: str = 'pad', padding_mode=0, **kwargs):
        super().__init__(target_multiple, method=method, **kwargs)
        self.padding_mode = padding_mode 
    
    def apply_transform(self, subject: Subject) -> Subject:
        source_shape = np.array(subject.spatial_shape, np.uint16)
        function: Callable = np.floor if self.method == 'crop' else np.ceil  # type: ignore[assignment]  # noqa: B950
        integer_ratio = function(source_shape / self.target_multiple)
        target_shape = integer_ratio * self.target_multiple
        target_shape = np.maximum(target_shape, 1)
        transform = tio.CropOrPad(target_shape.astype(int), padding_mode=self.padding_mode) ####### FIX #############
        subject = transform(subject)  # type: ignore[assignment]
        return subject

class CropOrPad(tio.CropOrPad):
    """CropOrPad. 
     random_center: Random center for crop and pad if no mask is set otherwise only random padding.
     center_along_slice_axis: When True, the last spatial axis (slice axis / D)
        is always center-cropped/padded, even if ``random_center`` randomizes
        the in-plane axes. Useful to keep the depth selection deterministic
        while still augmenting H/W."""

    def __init__(
        self,
        target_shape: Union[int, TypeTripletInt, None] = None,
        padding_mode: Union[str, float] = 0,
        mask_name: Optional[str] = None,
        labels: Optional[Sequence[int]] = None,
        random_center=False,
        center_along_slice_axis: bool = False,
        **kwargs,
    ):
        super().__init__(
            target_shape=target_shape,
            padding_mode=padding_mode,
            mask_name=mask_name,
            labels=labels,
            **kwargs
        )
        self.random_center = random_center
        self.center_along_slice_axis = center_along_slice_axis

    def _get_six_bounds_parameters(self, parameters: np.ndarray) :
        result = []
        n_axes = len(parameters)
        for axis_idx, number in enumerate(parameters):
            is_slice_axis = (axis_idx == n_axes - 1)
            randomize = self.random_center and not (self.center_along_slice_axis and is_slice_axis)
            if randomize:
                ini = np.random.randint(low=0, high=number+1)
            else:
                ini = int(np.ceil(number/2))
            fin = number-ini
            result.extend([ini, fin])
        return tuple(result)
    
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            if self.random_center:
                n_axes = len(padding_params) // 2
                random_padding_params = []
                for i in range(0, len(padding_params), 2):
                    axis_idx = i // 2
                    is_slice_axis = (axis_idx == n_axes - 1)
                    s = padding_params[i] + padding_params[i + 1]
                    if self.center_along_slice_axis and is_slice_axis:
                        r = int(np.ceil(s / 2))
                    else:
                        r = np.random.randint(0, s+1)
                    random_padding_params.extend([r, s - r])
                padding_params = random_padding_params
            pad = tio.Pad(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)  # type: ignore[assignment]
        return subject