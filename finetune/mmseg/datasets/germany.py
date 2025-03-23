# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from mmengine.logging import print_log
import pandas as pd
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
class GermanyCropDataset(BaseSegDataset):
    """Zurich dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    # {0: "unknown", 1: "sugar_beet", 2: "summer_oat", 3: "meadow", 5: "rape", 8: "hop",
    #                    9: "winter_spelt", 12: "winter_triticale", 13: "beans", 15: "peas", 16: "potatoes",
    #                    17: "soybeans", 19: "asparagus", 22: "winter_wheat", 23: "winter_barley", 24: "winter_rye",
    #                    25: "summer_barley", 26: "maize"}
    METAINFO = dict(
        classes=('sugar_beet', 'summer_oat', 'meadow', 'rape', 'hop', 'winter_spelt', 'winter_triticale', 'beans', 'peas',\
            'potatoes', 'soybeans', 'asparagus', 'winter_wheat', 'winter_barley', 'winter_rye', 'summer_barley', 'maize'),
        palette=[(255, 255, 255), (255, 255, 170), (255, 255, 85), (255, 170, 255), (255, 170, 170), (255, 170, 85), \
            (255, 85, 255), (255, 85, 170), (255, 85, 85), (170, 255, 255), (170, 255, 170), (170, 255, 85), (170, 170, 255), \
                 (170, 170, 170), (170, 170, 85), (170, 85, 255), (170, 85, 170)])
    def __init__(self,
                 img_suffix='.pickle',
                 seg_map_suffix='.pickle',
                 reduce_zero_label=True,
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
            print_log(f'dataset count: {len(lines)}')
            for line in lines:
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
