from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from packaging.version import Version
import numpy as np
from numpy import random
import math
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import mmcv
import copy
from mmcv.utils import deprecated_api_warning


class Compose(object):
    """Compose multiple transforms sequentially.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, sample):
        """Call function to apply transforms sequentially.

        Args:
            sample (Sample): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            sample = t(sample)
            if sample is None:
                return None
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class SegResize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in sample
        (img_scale = sample.img.shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long),
                                      max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short),
                                       max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, sample):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            sample (Sample): Sample data from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``sample``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = sample.img.shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        sample.scale = scale
        sample.scale_idx = scale_idx

    def _resize_img(self, sample):
        """Resize images with ``sample['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(sample[sample.img_field],
                                               sample.scale,
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = sample[sample.img_field].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(sample[sample.img_field],
                                                  sample.scale,
                                                  return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        sample[sample.img_field] = img
        sample.img_shape = img.shape
        sample.pad_shape = img.shape  # in case that there is no padding
        sample.scale_factor = scale_factor
        sample.keep_ratio = self.keep_ratio

    def _resize_seg(self, sample):
        """Resize semantic segmentation map with ``sample.scale``."""
        for key in sample.get('ann_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(sample[key],
                                        sample.scale,
                                        interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(sample[key],
                                       sample.scale,
                                       interpolation='nearest')
            sample[key] = gt_seg

    def __call__(self, sample):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            sample (Sample): Sample dict from loading pipeline.

        Returns:
            dict: Resized sample, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in sample:
            self._random_scale(sample)
        self._resize_img(sample)
        self._resize_seg(sample)
        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


class SegRandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='SegRandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, sample):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            sample (Sample): Sample data from loading pipeline.

        Returns:
            dict: Flipped sample, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in sample:
            flip = True if np.random.rand() < self.prob else False
            sample.flip = flip
        if 'flip_direction' not in sample:
            sample.flip_direction = self.direction
        if sample.flip:
            # flip image
            sample[sample.img_field] = mmcv.imflip(
                sample[sample.img_field], direction=sample.flip_direction)

            # flip segs
            for key in sample.get('ann_fields', []):
                # use copy() to make numpy stride positive
                sample[key] = mmcv.imflip(
                    sample[key], direction=sample.flip_direction).copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample):
        """Call function to normalize images.

        Args:
            sample (Sample): Sample data from loading pipeline.

        Returns:
            dict: Normalized sample, 'img_norm_cfg' key is added into
                sample.
        """

        sample[sample.img_field] = mmcv.imnormalize(sample[sample.img_field],
                                                    self.mean, self.std,
                                                    self.to_rgb)
        sample.img_norm_cfg = dict(mean=self.mean,
                                   std=self.std,
                                   to_rgb=self.to_rgb)
        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


class MSNormalize(object):

    def __init__(self, configs):
        self.configs = configs
        self.keys = configs.keys()

    def normalize_(self, img, config):
        if isinstance(img, np.ndarray) and img.dtype != np.float32:
            img = img.astype(np.float32)
        if isinstance(img, torch.Tensor):
            img = img.float()
        div_value = config.div_value
        mean = config.mean
        std = config.std
        img /= div_value
        for t, m, s in zip(img, mean, std):
            t -= m
            t /= s
        return img

    def __call__(self, sample):
        for key in self.keys:
            if isinstance(sample[key], list):
                for i in range(len(sample[key])):
                    sample[key][i] = self.normalize_(sample[key][i],
                                                     self.configs[key])
            else:
                sample[key] = self.normalize_(sample[key], self.configs[key])
        return sample


class MSRandomCrop(object):
    """Random crop the hr_img s2_img targets.
    Args:
        crop_size (tuple): Expected size ratio after cropping, (h, w).
    """

    def __init__(self, crop_size, keys):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.keys = keys

    def get_crop_bbox(self):
        """Randomly get a crop bounding box."""
        margin_h = max(1.0 - self.crop_size[0], 0)
        margin_w = max(1.0 - self.crop_size[1], 0)
        offset_h = np.random.uniform(0, margin_h)
        offset_w = np.random.uniform(0, margin_w)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        h, w = img.shape[-2:]
        crop_y1, crop_y2, crop_x1, crop_x2 = int(crop_y1 * h), int(
            crop_y2 * h), int(crop_x1 * w), int(crop_x2 * w)
        img = img[..., crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def __call__(self, sample):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        crop_bbox = self.get_crop_bbox()
        for key in self.keys:
            if isinstance(sample[key], list):
                for i in range(len(sample[key])):
                    sample[key][i] = self.crop(sample[key][i], crop_bbox)
            else:
                sample[key] = self.crop(sample[key], crop_bbox)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


class MSRandomRangeCrop(object):
    """Random crop the hr_img s2_img targets.
    Args:
        crop_size (tuple): Expected size ratio after cropping, (min, max).
    """

    def __init__(self, crop_size, keys):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.keys = keys

    def get_crop_bbox(self):
        """Randomly get a crop bounding box."""
        crop_size_ = np.random.uniform(self.crop_size[0], self.crop_size[1])
        margin_h = max(1.0 - crop_size_, 0)
        margin_w = max(1.0 - crop_size_, 0)
        offset_h = np.random.uniform(0, margin_h)
        offset_w = np.random.uniform(0, margin_w)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size_
        crop_x1, crop_x2 = offset_w, offset_w + crop_size_

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        h, w = img.shape[-2:]
        crop_y1, crop_y2, crop_x1, crop_x2 = int(crop_y1 * h), int(
            crop_y2 * h), int(crop_x1 * w), int(crop_x2 * w)
        img = img[..., crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def __call__(self, sample):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        crop_bbox = self.get_crop_bbox()
        for key in self.keys:
            if isinstance(sample[key], list):
                for i in range(len(sample[key])):
                    sample[key][i] = self.crop(sample[key][i], crop_bbox)
            else:
                sample[key] = self.crop(sample[key], crop_bbox)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


class MSResize(object):

    def __init__(self, target_size, keys):
        assert target_size[0] > 0 and target_size[1] > 0
        self.target_size = target_size
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key == 'targets':
                sample[key] = F.resize(
                    sample[key],
                    self.target_size,
                    interpolation=T.InterpolationMode.NEAREST
                    if Version(torchvision.__version__) >= Version('0.9.0')
                    else Image.NEAREST)
            else:
                sample[key] = F.resize(sample[key], self.target_size)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(target_size={self.target_size})'


class MSSSLRandomResizedCrop(object):

    def __init__(self, configs, global_crops_number, local_crops_number):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

    @staticmethod
    def get_params(scale: tuple, ratio: tuple):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect
                ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        origin_h, origin_w = 1.0, 1.0
        area = 1.0

        while True:
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = math.sqrt(target_area * aspect_ratio)
            h = math.sqrt(target_area / aspect_ratio)

            if w <= origin_w and h <= origin_h:
                i = random.uniform(0, origin_h - h)
                j = random.uniform(0, origin_w - w)
                return i, j, h, w

    def __call__(self, sample):
        for scope_view in self.configs.keys():
            for index in range(eval(f'self.{scope_view}_crops_number')):
                i, j, h, w = self.get_params(self.configs[scope_view].scale,
                                             self.configs[scope_view].ratio)
                for source in self.configs[scope_view]['size'].keys():
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    i_img, h_img = int(round(i * img.shape[-2])), int(
                        round(h * img.shape[-2]))
                    j_img, w_img = int(round(j * img.shape[-1])), int(
                        round(w * img.shape[-1]))
                    img = F.resized_crop(
                        img,
                        i_img,
                        j_img,
                        h_img,
                        w_img,
                        self.configs[scope_view]['size'][source],
                        interpolation=T.InterpolationMode.BICUBIC
                        if Version(torchvision.__version__) >= Version('0.9.0')
                        else Image.BICUBIC)
                    sample[img_key][index] = img
                    img_key = f'{scope_view}_{source}_distance'
                    img = sample[img_key][index]
                    img = F.resized_crop(
                        img,
                        i_img,
                        j_img,
                        h_img,
                        w_img,
                        self.configs[scope_view]['size'][source],
                        interpolation=T.InterpolationMode.BICUBIC
                        if Version(torchvision.__version__) >= Version('0.9.0')
                        else Image.BICUBIC)
                    sample[img_key][index] = img
                img_key = f'{scope_view}_lc'
                img = sample[img_key][index]
                i_img, h_img = int(round(i * img.shape[-2])), int(
                    round(h * img.shape[-2]))
                j_img, w_img = int(round(j * img.shape[-1])), int(
                    round(w * img.shape[-1]))
                img = F.resized_crop(
                    img,
                    i_img,
                    j_img,
                    h_img,
                    w_img,
                    self.configs[scope_view]['size']['s2'],
                    interpolation=T.InterpolationMode.NEAREST
                    if Version(torchvision.__version__) >= Version('0.9.0')
                    else Image.NEAREST)
                sample[img_key][index] = img
        return sample


class MSSSLRandomFlip(object):

    def __init__(self, configs, global_crops_number, local_crops_number,
                 scope_views, sources):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.scope_views = scope_views
        self.sources = sources

    def __call__(self, sample):
        for scope_view in self.scope_views:
            for index in range(eval(f'self.{scope_view}_crops_number')):
                hflip = False
                vflip = False
                for direction, prob in zip(self.configs['directions'],
                                           self.configs['probs']):
                    p = torch.rand(1)
                    if direction == 'horizontal' and p < prob:
                        hflip = True
                    if direction == 'vertical' and p < prob:
                        vflip = True
                for source in self.sources:
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    if hflip:
                        img = F.hflip(img)
                    if vflip:
                        img = F.vflip(img)
                    sample[img_key][index] = img
                    img_key = f'{scope_view}_{source}_distance'
                    img = sample[img_key][index]
                    if hflip:
                        img = F.hflip(img)
                    if vflip:
                        img = F.vflip(img)
                    sample[img_key][index] = img
                img_key = f'{scope_view}_lc'
                img = sample[img_key][index]
                if hflip:
                    img = F.hflip(img)
                if vflip:
                    img = F.vflip(img)
                sample[img_key][index] = img
        return sample


class MSSSLRandomRotate(object):

    def __init__(self, configs, global_crops_number, local_crops_number,
                 scope_views, sources):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.scope_views = scope_views
        self.sources = sources
        self.angle_set = [90, 180, 270]

    def __call__(self, sample):
        for scope_view in self.scope_views:
            for index in range(eval(f'self.{scope_view}_crops_number')):
                p = torch.rand(1)
                if p > self.configs['probs']:
                    continue
                angle = self.angle_set[torch.randint(0, 3, (1,)).item()]
                for source in self.sources:
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    img = F.rotate(
                        img,
                        angle,
                        interpolation=T.InterpolationMode.BILINEAR
                        if Version(torchvision.__version__) >= Version('0.9.0')
                        else Image.BILINEAR)
                    sample[img_key][index] = img
                    img_key = f'{scope_view}_{source}_distance'
                    img = sample[img_key][index]
                    img = F.rotate(
                        img,
                        angle,
                        interpolation=T.InterpolationMode.BILINEAR
                        if Version(torchvision.__version__) >= Version('0.9.0')
                        else Image.BILINEAR)
                    sample[img_key][index] = img
                img_key = f'{scope_view}_lc'
                img = sample[img_key][index]
                img = F.rotate(
                    img,
                    angle,
                    interpolation=T.InterpolationMode.NEAREST
                    if Version(torchvision.__version__) >= Version('0.9.0')
                    else Image.NEAREST)
                sample[img_key][index] = img
        return sample


class MSSSLRandomColorJitter(object):

    def __init__(self, configs, global_crops_number, local_crops_number,
                 scope_views, sources):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.scope_views = scope_views
        self.sources = sources
        self.color_prob = configs['color']['probs']
        self.brightness = configs['color']['brightness']
        self.contrast = configs['color']['contrast']
        self.saturation = configs['color']['saturation']
        self.hue = configs['color']['hue']
        self.gray_prob = configs['gray']['probs']

    def __call__(self, sample):
        for scope_view in self.scope_views:
            for index in range(eval(f'self.{scope_view}_crops_number')):
                for source in self.sources:
                    p = torch.rand(1)
                    if p >= self.color_prob:
                        continue
                    brightness_factor = random.uniform(
                        max(0, 1 - self.brightness), 1 + self.brightness)
                    contrast_factor = random.uniform(max(0, 1 - self.contrast),
                                                     1 + self.contrast)
                    saturation_factor = random.uniform(
                        max(0, 1 - self.saturation), 1 + self.saturation)
                    hue_factor = random.uniform(-self.hue, self.hue)
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    img = F.adjust_brightness(img, brightness_factor)
                    img = F.adjust_contrast(img, contrast_factor)
                    img = F.adjust_saturation(img, saturation_factor)
                    img = F.adjust_hue(img, hue_factor)
                    p = torch.rand(1)
                    if p >= self.gray_prob:
                        continue
                    num_output_channels, _, _ = img.shape
                    img = F.rgb_to_grayscale(
                        img, num_output_channels=num_output_channels)
                    sample[img_key][index] = img
        return sample


class MSSSLRandomGaussianBlur(object):

    def __init__(self, configs, global_crops_number, local_crops_number,
                 scope_views, sources):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.scope_views = scope_views
        self.sources = sources
        self.prob = configs['probs']
        self.sigma = configs['sigma']

    def __call__(self, sample):
        for scope_view in self.scope_views:
            for index in range(eval(f'self.{scope_view}_crops_number')):
                for source in self.sources:
                    p = self.prob[scope_view]
                    if scope_view == 'global':
                        p = p[index]
                    if torch.rand(1) >= p:
                        continue
                    sigma = random.uniform(self.sigma[0], self.sigma[1])
                    kernel_size = max(int(2 * ((sigma - 0.8) / 0.3 + 1) + 1),
                                      1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    img = F.gaussian_blur(img, kernel_size, sigma)
                    sample[img_key][index] = img
        return sample


class MSSSLRandomSolarize(object):

    def __init__(self, configs, global_crops_number, scope_views, sources):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.scope_views = scope_views
        self.sources = sources
        self.prob = configs['probs']
        self.threshold = 130

    def __call__(self, sample):
        for scope_view in self.scope_views:
            for index in range(eval(f'self.{scope_view}_crops_number')):
                for source in self.sources:
                    if index != 1:
                        continue
                    if torch.rand(1) >= self.prob:
                        continue
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    img = F.solarize(img, self.threshold)
                    sample[img_key][index] = img
        return sample


class MSSSLRandomChannelOut(object):

    def __init__(self, configs, global_crops_number, local_crops_number,
                 scope_views, sources, mean):
        self.configs = configs
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.scope_views = scope_views
        self.sources = sources
        self.mean = mean

    def __call__(self, sample):
        for scope_view in self.scope_views:
            for index in range(eval(f'self.{scope_view}_crops_number')):
                for source in self.sources:
                    out_num = self.configs[scope_view]['out_num']
                    out_num = random.randint(out_num[0], out_num[1] + 1)
                    out_index = sorted(
                        random.choice(len(self.mean), out_num, replace=False))
                    img_key = f'{scope_view}_{source}_img'
                    img = sample[img_key][index]
                    for i in out_index:
                        img[i] = int(self.mean[i])
                    sample[img_key][index] = img
        return sample


class MaskGenerator:

    def __init__(self,
                 input_size=192,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class DetResize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different sample. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long),
                                      max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short),
                                       max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, sample):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            sample (Sample): Result from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``sample``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        sample.scale = scale
        sample.scale_idx = scale_idx

    def _resize_img(self, sample):
        """Resize images with ``sample.scale``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                sample[sample.img_field],
                sample.scale,
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = sample[sample.img_field].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                sample[sample.img_field],
                sample.scale,
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend)
        sample[sample.img_field] = img

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        sample.img_shape = img.shape
        # in case that there is no padding
        sample.pad_shape = img.shape
        sample.scale_factor = scale_factor
        sample.keep_ratio = self.keep_ratio

    def _resize_bboxes(self, sample):
        """Resize bounding boxes with ``sample.scale_factor``."""
        for key in sample.get('bbox_fields', []):
            bboxes = sample[key] * sample.scale_factor
            if self.bbox_clip_border:
                img_shape = sample.img_shape
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            sample[key] = bboxes

    def _resize_masks(self, sample):
        """Resize masks with ``sample.scale``"""
        for key in sample.get('mask_fields', []):
            if sample[key] is None:
                continue
            if self.keep_ratio:
                sample[key] = sample[key].rescale(sample.scale)
            else:
                sample[key] = sample[key].resize(sample.img_shape[:2])

    def _resize_seg(self, sample):
        """Resize semantic segmentation map with ``sample.scale``."""
        for key in sample.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(sample[key],
                                        sample.scale,
                                        interpolation='nearest',
                                        backend=self.backend)
            else:
                gt_seg = mmcv.imresize(sample[key],
                                       sample.scale,
                                       interpolation='nearest',
                                       backend=self.backend)
            sample[key] = gt_seg

    def __call__(self, sample):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            sample (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized sample, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        sample.scale_idx = None
        if 'scale' not in sample:
            if 'scale_factor' in sample:
                img_shape = sample.hr_img.shape[:2]
                scale_factor = sample.scale_factor
                assert isinstance(scale_factor, float)
                sample.scale = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(sample)
        else:
            if not self.override:
                assert 'scale_factor' not in sample, (
                    'scale and scale_factor cannot be both set.')
            else:
                sample.pop('scale')
                if 'scale_factor' in sample:
                    sample.pop('scale_factor')
                self._random_scale(sample)

        self._resize_img(sample)
        self._resize_bboxes(sample)
        self._resize_masks(sample)
        self._resize_seg(sample)
        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


class DetRandomFlip:
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, sample):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            sample (Sample): Result from loading pipeline.

        Returns:
            Sample: Flipped sample, 'flip', 'flip_direction' keys are added \
                into sample.
        """

        if 'flip' not in sample:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            sample.flip = cur_dir is not None
        if 'flip_direction' not in sample:
            sample.flip_direction = cur_dir
        if sample.flip:
            # flip image
            sample[sample.img_field] = mmcv.imflip(
                sample[sample.img_field], direction=sample.flip_direction)
            # flip bboxes
            for key in sample.bbox_fields:
                sample[key] = self.bbox_flip(sample[key], sample.img_shape,
                                             sample.flip_direction)
            # flip masks
            for key in sample.mask_fields:
                sample[key] = sample[key].flip(sample.flip_direction)

            # flip segs
            for key in sample.seg_fields:
                sample[key] = mmcv.imflip(sample[key],
                                          direction=sample.flip_direction)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


class DetRandomCrop:
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped sample are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True):
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, sample, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            sample (Sample): Result from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped Sample data, 'img_shape' key in sample is
                updated according to crop size.
        """
        max_try_times = 20
        crop_times = 0
        while True:
            crop_times += 1
            assert crop_size[0] > 0 and crop_size[1] > 0
            img = sample[sample.img_field]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
            # crop bboxes accordingly and clip to the image boundary
            is_valid = False
            for key in sample.get('bbox_fields', []):
                # e.g. gt_bboxes and gt_bboxes_ignore
                bbox_offset = np.array(
                    [offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
                bboxes = sample[key] - bbox_offset
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, crop_size[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, crop_size[0])
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] >
                                                              bboxes[:, 1])
                sample[key] = bboxes[valid_inds, :]
                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in sample:
                    sample[label_key] = sample[label_key][valid_inds]
                # mask fields, e.g. gt_masks and gt_masks_ignore
                mask_key = self.bbox2mask.get(key)
                if mask_key in sample:
                    sample[mask_key] = sample[mask_key][
                        valid_inds.nonzero()[0]].crop(
                            np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                    if self.recompute_bbox:
                        sample[key] = sample[mask_key].get_bboxes()
                if valid_inds.any() and key == 'gt_bboxes':
                    is_valid = True
            if (crop_times
                    == max_try_times) or is_valid or allow_negative_crop:
                # crop the image
                img = sample[sample.img_field]
                img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
                img_shape = img.shape
                sample[sample.img_field] = img
                sample.img_shape = img_shape
                # crop semantic seg
                for key in sample.get('seg_fields', []):
                    sample[key] = sample[key][crop_y1:crop_y2, crop_x1:crop_x2]
                break
        return sample

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(min(h, self.crop_size[0]),
                                       min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(min(w, self.crop_size[0]),
                                       min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, sample):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            sample (Sample): Result from loading pipeline.

        Returns:
            Sample: Randomly cropped Sample data, 'img_shape' key in sample is
                updated according to crop size.
        """
        image_size = sample[sample.img_field].shape[:2]
        crop_size = self._get_crop_size(image_size)
        sample = self._crop_data(sample, crop_size, self.allow_negative_crop)
        sample.bbox_num = sample.gt_bboxes.shape[0]
        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


class AutoAugment:
    """Auto augmentation.

    This data augmentation is proposed in `Learning Data Augmentation
    Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_.

    TODO: Implement 'Shear', 'Sharpness' and 'Rotate' transforms

    Args:
        policies (list[list[transformer]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.
    """

    def __init__(self, policies):
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, sample):
        transform = np.random.choice(self.transforms)
        return transform(sample)
