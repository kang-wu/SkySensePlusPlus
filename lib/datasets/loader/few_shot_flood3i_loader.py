import os
import json
import datetime
import random
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

from antmmf.structures import Sample
from antmmf.datasets.base_dataset import BaseDataset
from antmmf.common import Configuration

from lib.datasets.utils.transforms import Compose, MSNormalize
from lib.datasets.utils.formatting import ToTensor
import lib.datasets.utils.pair_trainsforms as pair_transforms

from skimage import io
from osgeo import gdal
from PIL import Image


class FewShotFloodLoader(BaseDataset):
    DATASET_NAME = "few_shot_flood_loader"

    def __init__(self, dataset_type, config):
        super().__init__(self.__class__.DATASET_NAME, dataset_type, config)
        if dataset_type == 'train':
            raise ValueError('train mode not support!!!')

        self.root = config.data_root_dir
        self.dataset_type = dataset_type
        self.img_dir = config.img_dir
        self.tgt_dir = config.tgt_dir
        with open(config.data_txt, 'r') as f:
            test_list = f.readlines()
        self.test_pairs = []
        self.cls2path = {}
        for i in test_list:
            i = i.strip()
            if i == '':
                continue
            img_path = i[:-3]
            cls = int(i[-2:])
            cls = int(cls)
            self.test_pairs.append(
                {'hr_path': img_path,
                 'class': cls,
                 'tgt_path': img_path.replace('_', '_lab_', 1).replace('.jpg', '.png')
                 })
            if cls in self.cls2path.keys():
                self.cls2path[cls].append({'hr_path': img_path, 'tgt_path': img_path.replace('_', '_lab_', 1).replace('.jpg', '.png'), 'class': cls})
            else:
                self.cls2path[cls] = [{'hr_path': img_path, 'tgt_path': img_path.replace('_', '_lab_', 1).replace('.jpg', '.png'), 'class': cls}]

        self.seq_len = config.seq_len  # ts
        self.hr_size = config.image_size.hr
        self.s2_size = config.image_size.s2
        self.s1_size = config.image_size.s1
        self.anno_size = config.image_size.anno
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225])  # 先不管
        self.config = config
        self.pipeline = self._get_pipline()
        # self.crop_resize = pair_transforms.RandomResizedCropComb(512, scale=(0.99, 1.0), interpolation=3)

    def __len__(self) -> int:
        return len(self.test_pairs)

    def _combine_two_images(self, image, image2):
        dst = torch.cat([image, image2], dim=-2)
        return dst
    
    def _get_pipline(self):
        if self.dataset_type == 'val' or self.dataset_type == 'test':
            pipeline = [
                pair_transforms.ToTensor(),
                pair_transforms.RandomResizedCrop(512, scale=(0.9999, 1.0), interpolation=3),
                pair_transforms.Normalize(),
            ]
        else:
            raise ValueError('dataset_type not support')
        return pair_transforms.Compose(pipeline)

    def _load_data(self, data_path):
        file_name, file_extension = os.path.splitext(data_path)
        if file_extension == '.npz' or file_extension == '.npy':
            npz_key = self.config.get('npz_key', 'image')
            data = np.load(data_path)[npz_key]
        elif file_extension == '.png' or file_extension == '.jpg':
            data = io.imread(data_path)
            if len(data.shape) == 3:
                data = data.transpose(2, 0, 1)
        elif file_extension == '.tiff' or file_extension == '.tif':
            dataset = gdal.Open(data_path)
            if dataset is None:
                raise IOError(f'can not open file: {data_path}')
            data = dataset.ReadAsArray()
            dataset = None
        else:
            raise ValueError(f'file type {data_path} not support')
        # check nan
        if np.isnan(data).any():
            print(f'{data_path} with nan, replace it to 0!')
            data[np.isnan(data)] = 0
        return data

    def load_s2(self, pair):
        if 'l8_path' in pair.keys():
            pair['s2_path'] = pair['l8_path']

        if 's2_path' in pair.keys() and not self.config.get('masking_s2', False):
            with_s2 = True
            if isinstance(pair['s2_path'], list):
                if True:  # len(pair['s2_path']) > self.seq_len:
                    s2_path_list = np.random.choice(pair['s2_path'], self.seq_len)
                    s2_path_list = sorted(s2_path_list)
                else:
                    s2_path_list = pair['s2_path']
                s2_list = []
                s2_ct_1 = []
                for s2_path in s2_path_list:
                    s2 = self._load_data(os.path.join(self.root, s2_path))  # [:10]
                    s2_list.append(s2)
                    ct = os.path.splitext(s2_path)[0].split('_')
                    ct = ct[3]  # + ct[-3] + '01'
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
                if True:  # s2.shape[0] > self.seq_len:
                    selected_indices = np.random.choice(s2.shape[0], size=self.seq_len, replace=False)
                    selected_indices = sorted(selected_indices)
                    s2 = s2[selected_indices, :, :, :]
                    date = date[selected_indices]
                s2_1 = s2.transpose(1, 0, 2, 3)  # ts, c, h, w -> c, ts, h, w
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
                if True:  # len(pair['s1_path']) > self.seq_len:
                    s1_path_list = np.random.choice(pair['s1_path'], self.seq_len)
                    s1_path_list = sorted(s1_path_list)
                else:
                    s1_path_list = pair['s1_path']
                s1_list = []
                for s1_path in s1_path_list:
                    s1 = self._load_data(os.path.join(self.root, s1_path))
                    s1_list.append(s1)
                s1_1 = np.stack(s1_list, axis=1)
            else:
                s1 = self._load_data(os.path.join(self.root, pair['s1_path']))
                if True:  # s1.shape[0] > self.seq_len:
                    selected_indices = np.random.choice(s1.shape[0], size=self.seq_len, replace=False)
                    selected_indices = sorted(selected_indices)
                    s1 = s1[selected_indices, :, :, :]
                s1_1 = s1.transpose(1, 0, 2, 3)  # ts, c, h, w -> c, ts, h, w
        else:
            with_s1 = False
            s1_1 = np.zeros((2, self.seq_len, self.s1_size[0], self.s1_size[1]),
                            dtype=np.float32)
        return with_s1, s1_1

    def load_hr(self, pair):
        if 'hr_path' in pair.keys():
            with_hr = True
            hr = self._load_data(os.path.join(self.root, pair['hr_path']))
        else:
            with_hr = False
            hr = np.zeros((3, self.hr_size[0], self.hr_size[1]),
                           dtype=np.uint8)
        return with_hr, hr

    def load_tgt(self, pair):
        targets = self._load_data(os.path.join(self.root, pair['target_path']))
        return targets

    def get_item(self, idx):
        pair = self.test_pairs[idx]
        test_class = pair['class']

        current_dataset = 'flood3i'
        with_hr = True
        with_s2 = False
        with_s1 = False

        input_hr = io.imread(os.path.join(self.img_dir, pair['hr_path']))
        input_hr = input_hr.transpose(2,0,1)
        _, input_s2,_ = self.load_s2(pair)
        _, input_s1 = self.load_s1(pair)
        input_tgt = io.imread(os.path.join(self.tgt_dir, pair['tgt_path']))
        modality_dict = {
            's2': with_s2,
            's1': with_s1,
            'hr': with_hr
        }


        input_tgt[input_tgt != test_class] = 0
        input_tgt[input_tgt == test_class] = 255
        input_tgt = np.concatenate((input_tgt[None, :,:],)*3, axis=0)
        input_hr, input_s2, input_s1, input_tgt = self.pipeline(current_dataset, input_hr, input_s2, input_s1,
                                                                 input_tgt)

        while True:
            sel_prompt = random.choice(self.cls2path[test_class])
            if sel_prompt['hr_path'] != pair['hr_path']:
                break
        prompt_hr = io.imread(os.path.join(self.img_dir, sel_prompt['hr_path']))
        prompt_hr = prompt_hr.transpose(2,0,1)
        _, prompt_s2,_ = self.load_s2(pair)
        _, prompt_s1 = self.load_s1(pair)
        prompt_tgt = io.imread(os.path.join(self.tgt_dir, sel_prompt['tgt_path']))

        prompt_tgt[prompt_tgt != test_class] = 0
        prompt_tgt[prompt_tgt == test_class] = 255
        prompt_tgt = np.concatenate((prompt_tgt[None, :,:],)*3, axis=0)

        prompt_hr, prompt_s2, prompt_s1, prompt_tgt = self.pipeline(current_dataset, prompt_hr, prompt_s2, prompt_s1, prompt_tgt)

        targets_comb = self._combine_two_images(prompt_tgt, input_tgt)
        hr_comb = self._combine_two_images(prompt_hr, input_hr)
        s2_comb = self._combine_two_images(prompt_s2, input_s2)
        s1_comb = self._combine_two_images(prompt_s1, input_s1)

        valid = torch.ones_like(targets_comb)
        thres = torch.ones(3) * 1e-5  # ignore black
        thres = (thres - self.imagenet_mean) / self.imagenet_std
        valid[targets_comb < thres[:, None, None]] = 0

        mask_shape = (int(self.config.mim.input_size[0] / self.config.mim.patch_size),
                      int(self.config.mim.input_size[1] / self.config.mim.patch_size))
        mask = np.zeros(mask_shape, dtype=np.int32)
        mask[mask.shape[0] // 2:, :] = 1

        geo_location = pair["location"] if "location" in pair.keys() else None

        modality_idx = 2 ** 0 * modality_dict['s2'] + 2 ** 1 * modality_dict['s1'] + 2 ** 2 * modality_dict['hr']
        modality_flag_s2 = modality_dict['s2']
        modality_flag_s1 = modality_dict['s1']
        modality_flag_hr = modality_dict['hr']

        current_sample = Sample()
        current_sample.img_name = pair["tgt_path"].split('/')[-1].split('.')[0] + '-' +str(test_class)
        current_sample.hr_img = hr_comb
        current_sample.dataset_name = 'flood3i'
        current_sample.targets = targets_comb
        current_sample.s2_img = s2_comb
        current_sample.s2_ct = -1
        current_sample.s2_ct2 = -1
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
