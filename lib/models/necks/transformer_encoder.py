# Copyright (c) Ant Group. All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import (CheckpointLoader, load_state_dict)
from torch.nn.modules.batchnorm import _BatchNorm
from mmseg.models.backbones.vit import TransformerEncoderLayer

from mmseg.utils import get_root_logger


class TransformerEncoder(nn.Module):

    def __init__(self,
                 input_dims=768,
                 embed_dims=768,
                 num_layers=4,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(TransformerEncoder, self).__init__()

        self.porj_linear = nn.Linear(input_dims, embed_dims)
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        self.init_cfg = init_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims=embed_dims,
                                        num_heads=num_heads,
                                        feedforward_channels=mlp_ratio *
                                        embed_dims,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_rate=drop_rate,
                                        drop_path_rate=dpr[i],
                                        num_fcs=num_fcs,
                                        qkv_bias=qkv_bias,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg,
                                        with_cp=with_cp,
                                        batch_first=True))

    def init_weights(self):
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

            load_state_dict(self, state_dict, strict=False, logger=logger)
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward(self, inputs, require_feat: bool = False, require_two: bool = False):
        inputs = self.porj_linear(inputs)
        B, N, C = inputs.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, inputs), dim=1)
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        # add hidden and atten state
        block_outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if require_feat:
                block_outs.append(x)

        if self.output_cls_token:
            if require_two:
                x = x[:, :2]
            else:
                x = x[:, 0]
        elif not self.output_cls_token and self.with_cls_token:
            x = x # [:, :]

        if require_feat:
            return x, block_outs
        else:
            return x

    def train(self, mode=True):
        super(TransformerEncoder, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()