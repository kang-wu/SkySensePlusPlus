import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple
import numpy as np

import torch
from torch import Tensor
import torchvision.transforms as transforms
from skimage import io


try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from PIL import Image, ImageFilter, ImageOps

from .dataset_colors import modal_norm_dict

__all__ = [
    "Compose",
    "ToTensor",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
]



class Compose(transforms.Compose):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=None, interpolation2=None):
        # i = 0
        for t in self.transforms:
            # i = i+1
            # print(f'dataset_name:{dataset_name}')
            # print(f'step:{i}')
            # print(f'hr_img shape:{hr_img.shape}')
            # print(f's2_img shape:{s2_img.shape}')
            # print(f's1_img shape:{s1_img.shape}')
            # print(f'tgt shape:{tgt.shape}')
            
            hr_img, s2_img, s1_img, tgt = t(dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=interpolation1, interpolation2=interpolation2)
        return hr_img, s2_img, s1_img, tgt


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.
    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        

        # print(f'hr dtype:{hr_img.dtype}')
        # print(f's2_img dtype:{s2_img.dtype}')
        # print(f's1_img dtype:{s1_img.dtype}')
        # print(f'tgt dtype:{tgt.dtype}')
        if dataset_name == 'dynamic-mm' or dataset_name == 'guizhou-mm':
            hr_img = hr_img.astype(np.int32)[:3,:,:]
            hr_img = hr_img[::-1,:,:].copy()
        else:
            hr_img = hr_img.astype(np.int32)
        tgt = tgt.astype(np.uint8)
        s1_img = s1_img.astype(np.float32)
        s2_img = s2_img.astype(np.int16)
        
        return torch.tensor(hr_img), torch.tensor(s2_img), torch.tensor(s1_img),torch.tensor(tgt)


class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        # TODO 查询对应的mean和std
        
        # 处理一些mean和std
        if dataset_name == 'dynamic-mm':
            hr_std = [1008.4052, 760.9586, 631.4754]
            hr_mean = [1085.2941, 944.2718, 689.2493]
            hr_div = 1.
        else:
            hr_mean = modal_norm_dict['hr']['mean']
            hr_std = modal_norm_dict['hr']['std']
            hr_div = modal_norm_dict['hr']['div']            

        if dataset_name == 'l8activefire':
        # if False:
            s2_mean = modal_norm_dict['l8']['mean']
            s2_std = modal_norm_dict['l8']['std']
            s2_div = modal_norm_dict['l8']['div']
        else:
            s2_mean = modal_norm_dict['s2']['mean']
            s2_std = modal_norm_dict['s2']['std']
            s2_div = modal_norm_dict['s2']['div']
            
        s1_mean = modal_norm_dict['s1']['mean']
        s1_std = modal_norm_dict['s1']['std']
        s1_div = modal_norm_dict['s1']['div']

        anno_mean = [0.485, 0.456, 0.406]
        anno_std = [0.229, 0.224, 0.225]
        ann_div = 255.

        # 存在问题：时间序列这样处理是否会出错

        #mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        #std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        # print(s2_img.shape)
        # import pdb; pdb.set_trace()
        # print(s2_img)
        try:
            ch, ts, h, w = s2_img.shape
        except:
            print(f's2: {s2_img.shape}, s1: {s1_img.shape}')
        s2_img = s2_img.view(ch, ts*h, w)
        s2_img = self.normalize(s2_img.type(torch.float32), s2_mean, s2_std, self.inplace)
        s2_img = s2_img.view(ch, ts, h, w)

        ch, ts, h, w = s1_img.shape
        s1_img = s1_img.view(ch, ts*h, w)
        s1_img = self.normalize(s1_img.type(torch.float32), s1_mean, s1_std, self.inplace)
        s1_img = s1_img.view(ch, ts, h, w)

        # import pdb; pdb.set_trace()
        # print(s2_img.shape, s2_img[:,0,:,:])
        # print(s1_img.shape, s1_img[:,0,:,:])
        # print(hr_mean, hr_std, hr_div)
        return self.normalize(hr_img.type(torch.float32).div_(hr_div), hr_mean, hr_std, self.inplace), \
            s2_img, \
            s1_img, \
            self.normalize(tgt.type(torch.float32).div_(ann_div) , anno_mean, anno_std, self.inplace)

    def normalize(self, tensor, mean, std, inplace):
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        # print(f'tensor shape: {tensor.shape}')
        # print(f'mean shape: {mean.shape}')
        # print(f'std shape: {std.shape}')
        return tensor.sub_(mean).div_(std) 


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop a random portion of image and resize it to a given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.
    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        mode='small'
    ):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.cnt=0
        self.mode = mode

    def forward(self, dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=None, interpolation2=None, mode='small'):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(s2_img, self.scale, self.ratio)
        size_hr = hr_img.shape[-1]
        size_s2 = s2_img.shape[-1]
        size_anno = tgt.shape[-1]
        # 映射到其他模态
        ratio_s2_hr = size_s2 / size_hr
        i_hr = int(i / ratio_s2_hr)
        j_hr = int(j / ratio_s2_hr)
        h_hr = int(h / ratio_s2_hr)
        w_hr = int(w / ratio_s2_hr)

        ratio_s2_anno = size_s2 / size_anno
        i_anno = int(i / ratio_s2_anno)
        j_anno = int(j / ratio_s2_anno)
        h_anno = int(h / ratio_s2_anno)
        w_anno = int(w / ratio_s2_anno)

        if interpolation1 == 'nearest':
            interpolation1 = InterpolationMode.NEAREST
        else:
            interpolation1 = InterpolationMode.BICUBIC
        if interpolation2 == 'nearest':
            interpolation2 = InterpolationMode.NEAREST
        else:
            interpolation2 = InterpolationMode.BICUBIC
        # import pdb;pdb.set_trace()
        if self.scale[0]>0.99 and self.scale[0]<1.0:
            if self.mode=='small':
                resized_s2_img = F.resize(s2_img, (16,16), interpolation=InterpolationMode.BICUBIC)
                resized_hr_img = F.resize(hr_img, (512, 512), interpolation=InterpolationMode.BICUBIC)
                resized_s1_img = F.resize(s1_img, (16,16), interpolation=InterpolationMode.BICUBIC)
                resized_tgt = F.resize(tgt, (512,512), interpolation=InterpolationMode.NEAREST)
            else:
                resized_s2_img = F.resize(s2_img, (64,64), interpolation=InterpolationMode.BICUBIC)
                resized_hr_img = F.resize(hr_img, (2048, 2048), interpolation=InterpolationMode.BICUBIC)
                resized_s1_img = F.resize(s1_img, (64,64), interpolation=InterpolationMode.BICUBIC)
                resized_tgt = F.resize(tgt, (2048,2048), interpolation=InterpolationMode.NEAREST)
            return resized_hr_img, resized_s2_img, resized_s1_img, resized_tgt

        if self.mode=='small':
            resized_s2_img =  F.resized_crop(s2_img, i, j, h, w, (16, 16), InterpolationMode.BICUBIC)
            resized_hr_img = F.resized_crop(hr_img, i_hr, j_hr, h_hr, w_hr, (512, 512), InterpolationMode.BICUBIC)
            resized_s1_img = F.resized_crop(s1_img, i, j, h, w, (16, 16), InterpolationMode.BICUBIC)
            resized_tgt = F.resized_crop(tgt, i_anno, j_anno, h_anno, w_anno, (512, 512), InterpolationMode.NEAREST)
        else:
            resized_s2_img =  F.resized_crop(s2_img, i, j, h, w, (512, 512), InterpolationMode.BICUBIC)
            resized_hr_img = F.resized_crop(hr_img, i_hr, j_hr, h_hr, w_hr, (2048,2048), InterpolationMode.BICUBIC)
            resized_s1_img = F.resized_crop(s1_img, i, j, h, w, (512, 512), InterpolationMode.BICUBIC)
            resized_tgt = F.resized_crop(tgt, i_anno, j_anno, h_anno, w_anno, (2048, 2048), InterpolationMode.NEAREST)

        # import pdb; pdb.set_trace()
        # 将resize后的结果保存为concat的img
        # self.cnt = self.cnt+1
        # from torchvision.utils import save_image
        # save_hr = resized_hr_img[:3, :, :] / resized_hr_img[:3, :, :].max()
        # save_s2 = resized_s2_img[:3,0,:,:] / resized_s2_img[:3,0,:,:].max()
        # print(f'{save_hr.shape}, {save_s2.shape}')
        # save_image(save_s2, f'FoundationModel/debug/output2/resized_s2_{self.cnt}.png')
        # save_image(save_hr, f'FoundationModel/debug/output2/resized_hr_{self.cnt}.png')
        
        return resized_hr_img, resized_s2_img, resized_s1_img, resized_tgt
        
class RandomResizedCropComb(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.cnt=0

    def forward(self, dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(s2_img, self.scale, self.ratio)
        # print(f'i, j, h, w: {i, j, h, w}')
        # print(f's2_img shape: {s2_img.shape}')
        size_hr = hr_img.shape[-1]
        size_s2 = s2_img.shape[-1]
        size_anno = tgt.shape[-1]
        # 映射到其他模态
        ratio_s2_hr = size_s2 / size_hr
        i_hr = int(i / ratio_s2_hr)
        j_hr = int(j / ratio_s2_hr)
        h_hr = int(h / ratio_s2_hr)
        w_hr = int(w / ratio_s2_hr)

        ratio_s2_anno = size_s2 / size_anno
        i_anno = int(i / ratio_s2_anno)
        j_anno = int(j / ratio_s2_anno)
        h_anno = int(h / ratio_s2_anno)
        w_anno = int(w / ratio_s2_anno)

        if interpolation1 == 'nearest':
            interpolation1 = InterpolationMode.NEAREST
        else:
            interpolation1 = InterpolationMode.BICUBIC
        if interpolation2 == 'nearest':
            interpolation2 = InterpolationMode.NEAREST
        else:
            interpolation2 = InterpolationMode.BICUBIC
        
        resized_s2_img =  F.resized_crop(s2_img, i, j, h, w, (32, 16), InterpolationMode.BICUBIC)
        resized_hr_img = F.resized_crop(hr_img, i_hr, j_hr, h_hr, w_hr, (1024, 512), InterpolationMode.BICUBIC)
        resized_s1_img = F.resized_crop(s1_img, i, j, h, w, (32, 16), InterpolationMode.BICUBIC)
        resized_tgt = F.resized_crop(tgt, i_anno, j_anno, h_anno, w_anno, (1024, 512), InterpolationMode.NEAREST)
  
        return resized_hr_img, resized_s2_img, resized_s1_img, resized_tgt
        

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, dataset_name, hr_img, s2_img, s1_img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(hr_img), F.hflip(s2_img), F.hflip(s1_img), F.hflip(tgt)
        return hr_img, s2_img, s1_img, tgt


class RandomApply(transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability.
    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:
        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)
        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.
    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p=p)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        if self.p < torch.rand(1):
            return img, tgt
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt

class ColorJitter(transforms.ColorJitter):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (PIL Image or Tensor): Input image.
        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img, tgt


class RandomErasing(transforms.RandomErasing):
    """Randomly selects a rectangle region in a torch.Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    Returns:
        Erased Image.
    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None):
        """
        Args:
            img (Tensor): Tensor image to be erased.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return F.erase(img, x, y, h, w, v, self.inplace), tgt
        return img, tgt



class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, img, tgt, interpolation1=None, interpolation2=None):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img, tgt

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}( sigma={self.sigma})"
        return s

