import torch.nn as nn
from collections import OrderedDict
from mmcv.cnn.utils.weight_init import (kaiming_init, trunc_normal_)
from mmcv.runner import (CheckpointLoader, load_state_dict)
from mmseg.utils import get_root_logger


class UPHead(nn.Module):

    def __init__(self, in_dim, out_dim, up_scale, init_cfg=None):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=up_scale**2 * out_dim,
                      kernel_size=1),
            nn.PixelShuffle(up_scale),
        )
        self.init_cfg = init_cfg
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                _state_dict = checkpoint['state_dict']
            else:
                _state_dict = checkpoint

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            print(f'loading weight: {self.init_cfg["checkpoint"]}')
            load_state_dict(self, state_dict, strict=False, logger=logger)
        else:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', bias=0.)

    def forward(self, x):
        x = self.decoder(x)
        return x