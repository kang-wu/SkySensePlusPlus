import torch
import torch.nn as nn
from .multilevel_neck import MultiLevelNeck
from .fusion_transformer import FusionTransformer
from mmseg.registry import MODELS


@MODELS.register_module()
class FusionMultiLevelNeck(nn.Module):
    def __init__(self,
                 ts_size=10,
                 in_channels_ml=[768, 768, 768, 768],
                 out_channels_ml=768,
                 scales_ml=[0.5, 1, 2, 4],
                 norm_cfg_ml=None,
                 act_cfg_ml=None,
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
                 output_cls_token=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(FusionMultiLevelNeck, self).__init__()
        self.in_channels = in_channels_ml
        self.ts_size = ts_size
        self.multilevel_neck = MultiLevelNeck(
            in_channels_ml,
            out_channels_ml,
            scales_ml,
            norm_cfg_ml,
            act_cfg_ml
        )
        # self.up_head = UPHead(1024, 2816, 4)

        self.fusion_transformer = FusionTransformer(
            input_dims,
            embed_dims,
            num_layers,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            with_cls_token,
            output_cls_token,
            norm_cfg,
            act_cfg,
            num_fcs,
            norm_eval,
            with_cp,
            init_cfg,
        )
        
    def init_weights(self):
        self.fusion_transformer.init_weights()

    def forward(self, inputs, require_feat: bool = False, require_two: bool = False):
        assert len(inputs) == len(self.in_channels)

        inputs = self.multilevel_neck(inputs)
        
        ts = self.ts_size
        b_total, c, h, w = inputs[-1].shape
        b = int(b_total / ts)
        outs = []
        for idx in range(len(inputs)):

            input_feat = inputs[idx]
            b_total, c, h, w = inputs[idx].shape
            input_feat = input_feat.reshape(b, ts, c, h, w).permute(0, 3, 4, 1, 2).reshape(b*h*w, ts, c) # b*ts, c, h, w转换为b*h*w, ts, c
            feat_fusion = self.fusion_transformer(input_feat, require_feat, require_two)
            c_fusion = feat_fusion.shape[-1]
            feat_fusion = feat_fusion.reshape(b, h, w, c_fusion).permute(0, 3, 1, 2) # b*h*w, c -> b, c, h, w
            outs.append(feat_fusion)

        return tuple(outs)