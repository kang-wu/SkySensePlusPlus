import os
import json
import datetime
import random

import torch
import numpy as np
from osgeo import gdal
from skimage import io
from skimage.transform import resize

from antmmf.structures import Sample
from antmmf.datasets.base_dataset import BaseDataset

import lib.datasets.utils.pair_trainsforms as pair_transforms
from lib.datasets.utils.masking_generator import MaskingGenerator
from lib.datasets.utils.dataset_colors import dataset_color_dict, get_painter_color_map_list, get_real_random_color_list


class PretrainingLoader(BaseDataset):
    DATASET_NAME = "pretraining_loader"

    def __init__(self, dataset_type, config):
        super().__init__(self.__class__.DATASET_NAME, dataset_type, config)
        self.root = config.data_root_dir
        if dataset_type == 'train':
            self.json_path_list = config.train_json_path_list
        if dataset_type == 'val':
            self.json_path_list = config.val_json_path_list
        if dataset_type == 'test':
            self.json_path_list = config.val_json_path_list
        self.dataset_type = dataset_type
        self.pairs = []
        self.cls_repeat_cnt = config.cls_repeat_cnt
        num_datasets = len(self.json_path_list)
        for idx, json_path in enumerate(self.json_path_list):
            print(os.path.join(config.data_root_dir, json_path))
            cur_pairs = json.load(open(os.path.join(config.data_root_dir, json_path)))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
        
        if dataset_type == 'test' and config.prompt_json:
            cur_pairs = json.load(open(config.prompt_json))
            self.prompt = cur_pairs[0]
            print(f'prompt:{self.prompt}')
        
        self.use_multi_pairs = config.use_multi_pairs

        if self.use_multi_pairs:
            self.pair_type_dict = {}
            if dataset_type == 'train' or dataset_type == 'val':
                for idx, pair in enumerate(self.pairs):
                    if pair["type"] not in self.pair_type_dict:
                        new_subset = {}
                        classes = pair["classes"]
                        for cls in classes:
                            if cls not in new_subset.keys():
                                new_subset[cls] = [idx]
                            else:
                                new_subset[cls].append(idx)
                        self.pair_type_dict[pair["type"]] = new_subset
                    else:
                        classes = pair["classes"]
                        for cls in classes:
                            if cls not in self.pair_type_dict[pair["type"]].keys():
                                self.pair_type_dict[pair["type"]][cls] = [idx]
                            else:
                                self.pair_type_dict[pair["type"]][cls].append(idx)

        cnt = 0
        self.idx_to_cls = {}
        for k, v in self.pair_type_dict.items():
            for vv in v:
                self.idx_to_cls[cnt] = {
                    'type': k,
                    'classes_id': vv
                }
                cnt = cnt + 1

        print(self.idx_to_cls)
        self.idx_to_cls_list = []
        for i in self.idx_to_cls.keys():
            self.idx_to_cls_list.append(self.idx_to_cls[i])
        print(self.idx_to_cls_list)
        if self.dataset_type == 'train':
            self.idx_to_cls_list = self.idx_to_cls_list * self.cls_repeat_cnt
        self.masked_position_generator = MaskingGenerator(
            input_size=config.mim.input_size,
            patch_size=config.mim.patch_size,
            mask_ratio=config.mim.mask_ratio
        )
        if dataset_type == 'train':
            self.half_mask_ratio = config.half_mask_ratio
        else:
            self.half_mask_ratio = 1.

        self.seq_len = config.seq_len # ts
        self.hr_size = config.image_size.hr
        self.s2_size = config.image_size.s2
        self.s1_size = config.image_size.s1
        self.anno_size = config.image_size.anno
        self.min_random_scale = config.min_random_scale
        self.imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
        self.imagenet_std=torch.tensor([0.229, 0.224, 0.225])

        self.pipeline = self._get_pipline()
        self.crop_resize = pair_transforms.RandomResizedCropComb(512, scale=(0.3, 1.0), interpolation=3)
        self.num_samples = 8

    def __len__(self) -> int:
        return len(self.idx_to_cls_list)

    def _convert_colors_pairs(self, images, original_colors, new_colors, current_color):
        if len(original_colors) != len(new_colors):
            raise ValueError("The length of original_colors and new_colors must be the same.")
        unique_colors_list = []
        for image in images:
            if len(image.shape) == 3:
                image_hwc = image.transpose(1,2,0) # chw -> hwc 
            elif len(image.shape) == 2:
                image_hwc = image[:,:,None]
            else:
                raise ValueError('image shape is {image_hwc.shape}, which is not support to change color!')

            image_2d = image_hwc.reshape(-1, image_hwc.shape[-1])
            unique_colors = np.unique(image_2d, axis=0)
            unique_colors_list.append(unique_colors)
        unique_colors_list.append(original_colors)

        sets_of_tuples = [set(map(tuple, a)) for a in unique_colors_list]
        common_tuples = set.intersection(*sets_of_tuples)
        unique_old_colors = np.array(list(common_tuples), dtype=np.uint8)
        if len(unique_old_colors) == 0:
            unique_old_colors = [current_color]
        new_colors_coverted = new_colors[:len(unique_old_colors)]
        images_converted_list = []

        for image in images:
            image_convered = self._convert_colors(image, unique_old_colors, new_colors_coverted)
            images_converted_list.append(image_convered)

        return images_converted_list

    def _convert_colors(self, image, original_colors, new_colors):
        """
        Remap colors in an image to new colors.

        Parameters:
        image (numpy.ndarray): The image as a numpy array (channel x height x width).
        original_colors (list of tuples): The list of original colors to be replaced.
        new_colors (list of tuples): The list of new colors to replace the original colors.

        Returns:
        numpy.ndarray: The image with remapped colors. (channel x height x width)
        """

        if len(original_colors) != len(new_colors):
            raise ValueError("The length of original_colors and new_colors must be the same.")

        # Convert lists of tuples to numpy arrays for faster processing
        original_colors = np.array(original_colors)
        new_colors = np.array(new_colors)
        if len(original_colors.shape) == 1:
            original_colors = original_colors[:,None]
        
        # check image shape
        if len(image.shape) == 3:
            remapped_image = image.transpose(1,2,0) # chw -> hwc 
        elif len(image.shape) == 2:
            remapped_image = image[:,:,None]
        else:
            raise ValueError('image shape is {image.shape}, which is not support to change color!')
        
        # generate new image for return
        new_image = np.zeros((remapped_image.shape[0], remapped_image.shape[1], 3), dtype=np.uint8)

        for orig_color, new_color in zip(original_colors, new_colors):
            mask = np.all(remapped_image == orig_color, axis=-1)
            new_image[mask] = new_color

        new_image = new_image.transpose(2,0,1)  # hwc -> chw
        return new_image

    def _combine_images(self, images, interpolation='bicubic'):
        # images 8, c, h, w -> c, 4h, 2w
        group1 = images[:4]
        group2 = images[4:]
        stacked1 = torch.cat(group1, dim=-2)
        stacked2 = torch.cat(group2, dim=-2)
        result = torch.cat((stacked1, stacked2), dim=-1)

        return result

    def _get_pipline(self):
        if self.dataset_type == 'train':
            pipeline = [
                pair_transforms.ToTensor(),
                pair_transforms.RandomResizedCrop(512, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
                pair_transforms.RandomHorizontalFlip(),
                pair_transforms.Normalize(),
            ]
        elif self.dataset_type == 'val' or self.dataset_type == 'test':
            pipeline = [
                pair_transforms.ToTensor(),
                pair_transforms.RandomResizedCrop(512, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
                pair_transforms.Normalize(),
            ]
        else:
            raise ValueError('dataset_type not support')
        return pair_transforms.Compose(pipeline)

    def _load_data(self, data_path):
        file_name, file_extension = os.path.splitext(data_path)
        if file_extension == '.npz' or file_extension == '.npy':
            data = np.load(data_path)['image']
        elif file_extension == '.png' or file_extension == '.jpg':
            data = io.imread(data_path)
            if len(data.shape) == 3:
                data = data.transpose(2,0,1)
        elif file_extension == '.tiff' or file_extension == '.tif':
            dataset = gdal.Open(data_path)
            if dataset is None:
                raise IOError(f'无法打开文件{data_path}')
            data = dataset.ReadAsArray()
            dataset = None
        else:
            raise ValueError(f'file type {data_path} not support')
        if np.isnan(data).any():
            print(f'{data_path} with nan, replace it to 0!')
            data[np.isnan(data)] = 0
        return data

    def load_s2(self, pair):
        if pair['type'] == 'flair-mm' and 's2_path' in pair.keys():
            with_s2 =True
            s2 = np.load(os.path.join(self.root, pair['s2_path']))
            idx_centroid = pair['s2_cut_points']
            s2_patch_size = 40
            subset_sp = s2[:,:,idx_centroid[0]-int(s2_patch_size/2):idx_centroid[0] + \
                int(s2_patch_size/2),idx_centroid[1] - int(s2_patch_size/2):idx_centroid[1] + \
                int(s2_patch_size/2)]
            ts, c, h, w = subset_sp.shape
            subset_sp = subset_sp.reshape(-1, h, w).transpose(1,2,0)
            s2 = resize(subset_sp, (16, 16), anti_aliasing=True).transpose(2,0,1)
            s2 = s2.reshape(ts, c, 16, 16)
            if True:
                selected_indices = np.random.choice(s2.shape[0], size=self.seq_len, replace=False)
                selected_indices = sorted(selected_indices)
                s2 = s2[selected_indices, :, :, :]
            
            s2_1 = s2.transpose(1,0,2,3) # ts, c, h, w -> c, ts, h, w
            s2_ct_1 = [0] * self.seq_len

        elif 's2_path' in pair.keys():
            with_s2 =True
            if isinstance(pair['s2_path'], list):
                if True:
                    s2_path_list = np.random.choice(pair['s2_path'], self.seq_len)
                    s2_path_list = sorted(s2_path_list)
                else:
                    s2_path_list = pair['s2_path']
                s2_list = []
                s2_ct_1 = []
                for s2_path in s2_path_list:
                    s2 = self._load_data(os.path.join(self.root, s2_path))#[:10]
                    s2_list.append(s2)
                    ct = os.path.splitext(s2_path)[0].split('_')
                    ct = ct[-4] + ct[-3] + '01'
                    try:
                        ct = datetime.datetime.strptime(ct, '%Y%m%d')
                    except:
                        ct = datetime.datetime.strptime(ct, '%Y-%m-%d')
                    ct = ct.timetuple()
                    ct = ct.tm_yday - 1
                    s2_ct_1.append(ct)
                s2_1 = np.stack(s2_list, axis=1) 
                
            else: 
                s2 = np.load(os.path.join(self.root, pair['s2_path']))['image']
                date = np.load(os.path.join(self.root, pair['s2_path']))['date']
                if True:
                    selected_indices = np.random.choice(s2.shape[0], size=self.seq_len, replace=False)
                    selected_indices = sorted(selected_indices)
                    s2 = s2[selected_indices, :, :, :]
                    date = date[selected_indices]
                s2_1 = s2.transpose(1,0,2,3) # ts, c, h, w -> c, ts, h, w
                s2_ct_1 = []
                for ct in date:
                    try:
                        ct = datetime.datetime.strptime(ct, '%Y%m%d')
                    except:
                        ct = datetime.datetime.strptime(ct, '%Y-%m-%d')
                    ct = ct.timetuple()
                    ct = ct.tm_yday - 1
                    s2_ct_1.append(ct)
        else:
            with_s2 = False
            s2_1 = np.zeros((10, self.seq_len, self.s2_size[0], self.s2_size[1]),
                           dtype=np.int16)
            s2_ct_1 = [0] * self.seq_len
        
        return with_s2, s2_1, s2_ct_1

    def load_s1(self, pair):
        if 's1_path' in pair.keys():
            with_s1 = True
            if isinstance(pair['s1_path'], list):
                if True:
                    s1_path_list = np.random.choice(pair['s1_path'], self.seq_len)
                    s1_path_list = sorted(s1_path_list)
                else:
                    s1_path_list = pair['s1_path']
                s1_list = []
                for s1_path in s1_path_list:
                    s1 =  self._load_data(os.path.join(self.root, s1_path))
                    s1_list.append(s1)
                s1_1 = np.stack(s1_list, axis=1)
            else:
                s1 = self._load_data(os.path.join(self.root, pair['s1_path']))
                if True:
                    selected_indices = np.random.choice(s1.shape[0], size=self.seq_len, replace=False)
                    selected_indices = sorted(selected_indices)
                    s1 = s1[selected_indices, :, :, :]
                s1_1 = s1.transpose(1,0,2,3) # ts, c, h, w -> c, ts, h, w
        else:
            with_s1 = False
            s1_1 = np.zeros((2, self.seq_len, self.s1_size[0], self.s1_size[1]),
                           dtype=np.float32)
        return with_s1, s1_1
    
    def load_hr(self, pair):
        if 'hr_path' in pair.keys():
            if pair['type'] == 'flair-mm':
                with_hr = True
                hr = self._load_data(os.path.join(self.root, pair['hr_path']))[:3,:,:]
            else:
                with_hr = True
                hr = self._load_data(os.path.join(self.root, pair['hr_path']))
        else:
            with_hr = False
            hr = np.zeros((3, self.hr_size[0], self.hr_size[1]),
                           dtype=np.uint8)
        return with_hr, hr

    def load_tgt(self, pair):
        if self.dataset_type == 'test':
            targets = np.zeros((3, self.anno_size[0], self.anno_size[1]),
                           dtype=np.uint8)
        else:
            targets = self._load_data(os.path.join(self.root, pair['target_path']))
        return targets

    def find_random_position(self, matrix, current_color):
        if matrix.ndim == 2:
            matrix = matrix[None, :, :]
        current_color = np.array(current_color)
        C, H, W = matrix.shape
        
        if len(current_color) != C:
            raise ValueError("current_color unmatch with matrix!")
        
        matches = np.where(np.all(matrix == current_color[:, None, None], axis=0))

        if len(matches[0]) > 0:
            index = np.random.choice(range(len(matches[0])))
            return (matches[0][index], matches[1][index])
        else:
            return None

    def get_item(self, idx):
        dataset_cls_infos = self.idx_to_cls_list[idx]
        current_dataset = dataset_cls_infos['type']
        current_classes_id = dataset_cls_infos['classes_id']
        pair_idx_list = self.pair_type_dict[current_dataset][current_classes_id]

        old_colors = dataset_color_dict[current_dataset]
        current_color = old_colors[current_classes_id]
        class_num = len(old_colors)
        if self.dataset_type == 'train':
            new_colors = get_real_random_color_list(class_num)
        else:
            new_colors = get_painter_color_map_list(class_num) # fix colors mapping when testing

        num_samples = self.num_samples
        if len(pair_idx_list) < num_samples:
            selected_samples = [random.choice(pair_idx_list) for _ in range(num_samples)]
        else:
            selected_samples = random.sample(pair_idx_list, num_samples)
        hr_imgs = []
        tgt_imgs = []
        s2_imgs = []
        s1_imgs = []
        s2_cts = []
        for sample_idx in selected_samples:
            pair = self.pairs[sample_idx]
            with_hr, hr = self.load_hr(pair)
            with_s2, s2, s2_ct_1 = self.load_s2(pair)
            with_s1, s1 = self.load_s1(pair)
            tgt = self.load_tgt(pair)
            modality_dict = {
                's2' : with_s2,
                's1' : with_s1,
                'hr' : with_hr
            }

            if (hr.shape[-2:] != tuple(self.hr_size)) and (hr.shape[-2:] == tgt.shape[-2:]) and (self.hr_size == self.anno_size):
                point_pos = self.find_random_position(tgt, current_color)
                upper_left_raw = [point_pos[0] - self.hr_size[0] // 2, point_pos[1] - self.hr_size[1] // 2]
                upper_left = [i - i%32 + 16 for i in upper_left_raw]
                upper_left_sentinel = [i // 32 for i in upper_left_raw]
                upper_left[0] = np.clip(np.array(upper_left[0]), 0, hr.shape[-2] - self.hr_size[0])
                upper_left[1] = np.clip(np.array(upper_left[1]), 0, hr.shape[-1] - self.hr_size[1])

                upper_left_sentinel[0] = np.clip(np.array(upper_left_sentinel[0]), 0, s1.shape[-2] - self.s1_size[0])
                upper_left_sentinel[1] = np.clip(np.array(upper_left_sentinel[1]), 0, s1.shape[-1] - self.s1_size[1])
                hr = hr[:, upper_left[0]:upper_left[0]+self.hr_size[0], upper_left[1]:upper_left[1]+self.hr_size[1]]
                if with_s1:
                    s1 = s1[:, :, upper_left_sentinel[0]:upper_left_sentinel[0]+self.s1_size[0], upper_left_sentinel[1]:upper_left_sentinel[1]+self.s1_size[1]]
                if with_s2:
                    s2 = s2[:, :, upper_left_sentinel[0]:upper_left_sentinel[0]+self.s2_size[0], upper_left_sentinel[1]:upper_left_sentinel[1]+self.s2_size[1]]
                if tgt.ndim == 3:
                    tgt = tgt[:, upper_left[0]:upper_left[0]+self.hr_size[0], upper_left[1]:upper_left[1]+self.hr_size[1]]
                elif tgt.ndim == 2:
                    tgt = tgt[upper_left[0]:upper_left[0]+self.hr_size[0], upper_left[1]:upper_left[1]+self.hr_size[1]]
                else:
                    raise ValueError("tgt dim unsupport!")
            hr_imgs.append(hr)
            tgt_imgs.append(tgt)
            s2_imgs.append(s2)
            s1_imgs.append(s1)
            s2_cts.append(s2_ct_1)
        
        
        cvt_hr_imgs = []
        cvt_tgt_imgs = []
        cvt_s2_imgs = []
        cvt_s1_imgs = []

        tgt_imgs = self._convert_colors_pairs(tgt_imgs, old_colors, new_colors, current_color)
        for i in range(len(tgt_imgs)):
            hr, s2, s1, tgt = self.pipeline(current_dataset, hr_imgs[i], s2_imgs[i], s1_imgs[i], tgt_imgs[i])
            cvt_hr_imgs.append(hr)
            cvt_s2_imgs.append(s2)
            cvt_s1_imgs.append(s1)
            cvt_tgt_imgs.append(tgt)

        targets_comb = self._combine_images(cvt_tgt_imgs)
        hr_comb = self._combine_images(cvt_hr_imgs)
        s2_comb = self._combine_images(cvt_s2_imgs)
        s1_comb = self._combine_images(cvt_s1_imgs)
        hr_comb, s2_comb, s1_comb, targets_comb = self.crop_resize(current_dataset, hr_comb, s2_comb, s1_comb, targets_comb)
        use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
        valid = torch.ones_like(targets_comb)

        thres = torch.ones(3) * (1e-5) # ignore black
        thres = (thres - self.imagenet_mean) / self.imagenet_std
        valid[targets_comb < thres[:, None, None]] = 0
        
        if use_half_mask:
            num_patches = self.masked_position_generator.num_patches
            mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
            mask[mask.shape[0]//2:, :] = 1
        else:
            mask = self.masked_position_generator()
        
        # location
        geo_location = pair["location"] if "location" in pair.keys() else None

        # get modality index
        modality_idx = 2**0 * modality_dict['s2'] + 2**1 * modality_dict['s1'] + 2**2 * modality_dict['hr']
        modality_flag_s2 = modality_dict['s2']
        modality_flag_s1 = modality_dict['s1']
        modality_flag_hr = modality_dict['hr']


        current_sample = Sample()
        current_sample.img_name = pair["hr_path"].split('/')[-1].split('.')[0]
        current_sample.hr_img = hr_comb
        current_sample.dataset_name = pair["type"]
        current_sample.targets = targets_comb
        current_sample.s2_img = s2_comb
        current_sample.s2_ct = s2_cts[0]
        current_sample.s2_ct2 = s2_cts[4]
        current_sample.s1_img = s1_comb
        current_sample.anno_mask = torch.from_numpy(mask)
        current_sample.valid = valid
        current_sample.location = geo_location
        current_sample.modality_idx = modality_idx
        current_sample.modality_flag_s2 = modality_flag_s2
        current_sample.modality_flag_s1 = modality_flag_s1
        current_sample.modality_flag_hr = modality_flag_hr
        current_sample.task_type = self.dataset_type
        
        return current_sample