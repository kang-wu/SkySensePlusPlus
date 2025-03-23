# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

# LEGEND = [
#     255    255    255;  % Background
#       0      0      0;  % Roads
#     100    100    100;  % Buildings
#       0    125      0;  % Trees
#       0    255      0;  % Grass
#     150     80      0;  % Bare Soil
#       0      0    150;  % Water
#     255    255      0;  % Railways
#     150    150    255]; % Swimming Pools 

@DATASETS.register_module()
class C2SFloodDataset(BaseSegDataset):
    """Zurich dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('Background', 'Water', 'Cloud', 'Cloud shadow'),
        palette=[[0,0,0], [255,255,255], [255,0,0], [0,255,0]]
    )
    def __init__(self,
                 img_suffix='.npz',
                 seg_map_suffix='.npz',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
    
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = json.load(open(self.ann_file))
            for line in lines:
                # img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(self.data_root, line['s2_path']))
                if 'target_path' in line.keys():
                    data_info['seg_map_path'] = osp.join(self.data_root, line['target_path'])
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
