# coding: utf-8
# Copyright (c) Ant Group. All rights reserved.
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import math
import random
from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from lib.models.backbones import build_backbone
from lib.models.necks import build_neck
from lib.models.heads import build_head
from lib.utils.utils import LayerDecayValueAssigner


@registry.register_model("SkySensePP")
class SkySensePP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sources = config.sources
        assert len(self.sources) > 0, 'at least one data source is required'
        if 's2' in self.sources:
            self.use_ctpe = config.use_ctpe
        self.use_modal_vae = config.use_modal_vae
        self.use_cls_token_uper_head = config.use_cls_token_uper_head
        self.target_mean=[0.485, 0.456, 0.406]
        self.target_std=[0.229, 0.224, 0.225]
        self.vocabulary_size = config.vocabulary_size
        self.vocabulary = list(range(1, config.vocabulary_size + 1)) # 0 for ignore
        

    def build(self):
        if 'hr' in self.sources:
            self.backbone_hr = self._build_backbone('hr')
        if 's2' in self.sources:
            self.backbone_s2 = self._build_backbone('s2')
            if self.use_ctpe:
                self.ctpe = nn.Parameter(
                    torch.zeros(1, self.config.calendar_time,
                                self.config.necks.input_dims))
            if 'head_s2' in self.config.keys():
                self.head_s2 = self._build_head('head_s2')
            self.fusion = self._build_neck('necks')
        if 's1' in self.sources:
            self.backbone_s1 = self._build_backbone('s1')
            if 'head_s1' in self.config.keys():
                self.head_s1 = self._build_head('head_s1')
        self.head_rec_hr = self._build_head('rec_head_hr')

        self.with_aux_head = False
        
        if self.use_modal_vae:
            self.modality_vae = self._build_neck('modality_vae')
        if 'auxiliary_head' in self.config.keys():
            self.with_aux_head = True
            self.aux_head = self._build_head('auxiliary_head')
        if 'init_cfg' in self.config.keys(
        ) and self.config.init_cfg is not None and self.config.init_cfg.checkpoint is not None and self.config.init_cfg.key is not None:
            self.load_pretrained(self.config.init_cfg.checkpoint,
                                 self.config.init_cfg.key)

    def _build_backbone(self, key):
        config_dict = self.config[f'backbone_{key}'].to_dict()
        backbone_type = config_dict.pop('type')
        backbone = build_backbone(backbone_type, **config_dict)
        backbone.init_weights()
        return backbone

    def _build_neck(self, key):
        config_dict = self.config[key].to_dict()
        neck_type = config_dict.pop('type')
        neck = build_neck(neck_type, **config_dict)
        neck.init_weights()
        return neck

    def _build_head(self, key):
        head_config = self.config[key].to_dict()
        head_type = head_config.pop('type')
        head = build_head(head_type, **head_config)
        return head

    def get_optimizer_parameters(self, config):
        optimizer_grouped_parameters = [
            {
                "params": [],
                "lr": config.optimizer_attributes.params.lr,
                "weight_decay": config.optimizer_attributes.params.weight_decay,
            },
            {
                "params": [],
                "lr": config.optimizer_attributes.params.lr,
                "weight_decay": 0.0,
            },
        ]
        layer_decay_value_assigner_hr = LayerDecayValueAssigner(
            config.lr_parameters.layer_decay, None,
            config.optimizer_attributes.params.lr, 'swin', 
            config.model_attributes.SkySensePP.backbone_hr.arch
        )
        layer_decay_value_assigner_s2 = LayerDecayValueAssigner(
            config.lr_parameters.layer_decay, 24,
            config.optimizer_attributes.params.lr, 'vit', 
        )
        layer_decay_value_assigner_s1 = LayerDecayValueAssigner(
            config.lr_parameters.layer_decay, 24,
            config.optimizer_attributes.params.lr, 'vit', 
        )
        layer_decay_value_assigner_fusion = LayerDecayValueAssigner(
            config.lr_parameters.layer_decay, 24,
            config.optimizer_attributes.params.lr, 'vit', 
        )
        num_frozen_params = 0
        if 'hr' in self.sources:
            print('hr'.center(60, '-'))
            num_frozen_params += layer_decay_value_assigner_hr.fix_param(
                self.backbone_hr, 
                config.lr_parameters.frozen_blocks,
            )
            optimizer_grouped_parameters.extend(
                layer_decay_value_assigner_hr.get_parameter_groups(
                    self.backbone_hr, config.optimizer_attributes.params.weight_decay
                )
            )
        if 's2' in self.sources:
            print('s2'.center(60, '-'))
            num_frozen_params += layer_decay_value_assigner_s2.fix_param(
                self.backbone_s2, 
                config.lr_parameters.frozen_blocks,
            )
            optimizer_grouped_parameters.extend(
                layer_decay_value_assigner_s2.get_parameter_groups(
                    self.backbone_s2, config.optimizer_attributes.params.weight_decay
                )
            )
            no_decay = [".bn.", "bias"]
            optimizer_grouped_parameters[0]["params"] += [
                p for n, p in self.head_s2.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
            optimizer_grouped_parameters[1]["params"] += [
                p for n, p in self.head_s2.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
            if self.use_ctpe:
                optimizer_grouped_parameters[1]["params"] += [self.ctpe]

        if 's1' in self.sources:
            print('s1'.center(60, '-'))
            num_frozen_params += layer_decay_value_assigner_s1.fix_param(
                self.backbone_s1, 
                config.lr_parameters.frozen_blocks,
            )
            optimizer_grouped_parameters.extend(
                layer_decay_value_assigner_s1.get_parameter_groups(
                    self.backbone_s1, config.optimizer_attributes.params.weight_decay
                )
            )
            no_decay = [".bn.", "bias"]
            optimizer_grouped_parameters[0]["params"] += [
                p for n, p in self.head_s1.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
            optimizer_grouped_parameters[1]["params"] += [
                p for n, p in self.head_s1.named_parameters()
                if any(nd in n for nd in no_decay)
            ]

        if len(self.sources) > 1:
            print('fusion'.center(60, '-'))
            num_frozen_params += layer_decay_value_assigner_fusion.fix_param_deeper(
                self.fusion, 
                config.lr_parameters.frozen_fusion_blocks_start, # 冻结后面所有的stage
            )
            optimizer_grouped_parameters.extend(
                layer_decay_value_assigner_fusion.get_parameter_groups(
                    self.fusion, config.optimizer_attributes.params.weight_decay
                )
            )

        if self.use_modal_vae:
            no_decay = [".bn.", "bias"]
            optimizer_grouped_parameters[0]["params"] += [
                p for n, p in self.modality_vae.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
            optimizer_grouped_parameters[1]["params"] += [
                p for n, p in self.modality_vae.named_parameters()
                if any(nd in n for nd in no_decay)
            ]

        no_decay = [".bn.", "bias"]
        optimizer_grouped_parameters[0]["params"] += [
            p for n, p in self.head_rec_hr.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        optimizer_grouped_parameters[1]["params"] += [
            p for n, p in self.head_rec_hr.named_parameters()
            if any(nd in n for nd in no_decay)
        ]

        if self.with_aux_head:
            no_decay = [".bn.", "bias"]
            optimizer_grouped_parameters[0]["params"] += [
                p for n, p in self.aux_head.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
            optimizer_grouped_parameters[1]["params"] += [
                p for n, p in self.aux_head.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
        num_params = [len(x['params']) for x in optimizer_grouped_parameters]
        print(len(list(self.parameters())), sum(num_params), num_frozen_params)
        assert len(list(self.parameters())) == sum(num_params) + num_frozen_params 
        return optimizer_grouped_parameters

    def get_custom_scheduler(self, trainer):
        optimizer = trainer.optimizer
        num_training_steps = trainer.config.training_parameters.max_iterations
        num_warmup_steps = trainer.config.training_parameters.num_warmup_steps

        if "train" in trainer.run_type:
            if num_training_steps == math.inf:
                epoches = trainer.config.training_parameters.max_epochs
                assert epoches != math.inf
                num_training_steps = trainer.config.training_parameters.max_epochs * trainer.epoch_iterations

            def linear_with_wram_up(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(
                        1, num_warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step) /
                    float(max(1, num_training_steps - num_warmup_steps)),
                )

            def cos_with_wram_up(current_step):
                num_cycles = 0.5
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(
                        1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps))
                return max(
                    0.0,
                    0.5 *
                    (1.0 +
                     math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
                )

            lr_lambda = cos_with_wram_up if trainer.config.training_parameters.cos_lr else linear_with_wram_up

        else:
            def lr_lambda(current_step):
                return 0.0  # noqa

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)

    def convert_target(self, target):
        mean = target.new_tensor(self.target_mean).reshape(1, 3, 1, 1)
        std = target.new_tensor(self.target_std).reshape(1, 3, 1, 1)
        target = ((target * std + mean)*255).to(torch.long)
        target[:, 0] = target[:, 0] * 256 * 256
        target[:, 1] = target[:, 1] * 256
        target = target.sum(1).type(torch.long)
        unique_target = target.unique()
        target_index = torch.searchsorted(unique_target, target)
        no_bg = False
        if unique_target[0].item() > 0:
            target_index += 1
            no_bg = True
        target_index_unique = target_index.unique().tolist()
        random.shuffle(self.vocabulary)
        value = target.new_tensor([0] + self.vocabulary)
        mapped_target = target_index.clone()
        idx_2_color = {}
        for v in target_index_unique:
            mapped_target[target_index == v] = value[v]
            idx_2_color[value[v].item()] = unique_target[v - 1 if no_bg else v].item()
        return mapped_target, idx_2_color

    def forward(self, sample_list):
        output = dict()
        modality_flag_hr = sample_list["modality_flag_hr"]
        modality_flag_s2 = sample_list["modality_flag_s2"]
        modality_flag_s1 = sample_list["modality_flag_s1"]
        modalities = [modality_flag_hr, modality_flag_s2, modality_flag_s1]
        modalities = torch.tensor(modalities).permute(1,0).contiguous() # L, B => B, L

        anno_img = sample_list["targets"]
        anno_img, idx_2_color = self.convert_target(anno_img)
        output["mapped_targets"] = anno_img
        output["idx_2_color"] = idx_2_color
        anno_mask = sample_list["anno_mask"]
        anno_s2 = anno_img[:, 15::32, 15::32]
        anno_s1 = anno_s2

        output["anno_hr"] = anno_img
        output["anno_s2"] = anno_s2
        
        ### 1. backbone
        if 'hr' in self.sources:
            hr_img = sample_list["hr_img"]
            B_MASK, H_MASK, W_MASK = anno_mask.shape
            block_size = 32
            anno_mask_hr = anno_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, block_size, block_size)
            anno_mask_hr = anno_mask_hr.permute(0, 1, 3, 2, 4).reshape(B_MASK, H_MASK*block_size, W_MASK*block_size).contiguous()
            B, C_G, H_G, W_G = hr_img.shape
            hr_features = self.backbone_hr(hr_img, anno_img, anno_mask_hr)
            output['mask_hr'] = anno_mask_hr
            output['target_hr'] = anno_img
            
        if 's2' in self.sources:
            s2_img = sample_list["s2_img"]
            B, C_S2, S_S2, H_S2, W_S2 = s2_img.shape
            s2_img = s2_img.permute(0, 2, 1, 3,
                                    4).reshape(B * S_S2, C_S2, H_S2, W_S2).contiguous() # ts time to batch
            anno_mask_s2 = anno_mask
            s2_features = self.backbone_s2(s2_img, anno_s2, anno_mask_s2)
            if 'head_s2' in self.config.keys():
                s2_features = self.head_s2(s2_features[-1])
                s2_features = [s2_features]

        if 's1' in self.sources:
            s1_img = sample_list["s1_img"]
            B, C_S1, S_S1, H_S1, W_S1 = s1_img.shape
            s1_img = s1_img.permute(0, 2, 1, 3,
                                    4).reshape(B * S_S1, C_S1, H_S1, W_S1).contiguous()
    
            anno_mask_s1 = anno_mask
            s1_features = self.backbone_s1(s1_img, anno_s1, anno_mask_s1)
            if 'head_s1' in self.config.keys():
                s1_features = self.head_s1(s1_features[-1])
                s1_features = [s1_features]

        ### 2. prepare features for fusion
        hr_features_stage3 = hr_features[-1]
        s2_features_stage3 = s2_features[-1]
        s1_features_stage3 = s1_features[-1]
        modalities = modalities.to(hr_features_stage3.device)
        if self.use_modal_vae:
            vae_out = self.modality_vae(hr_features_stage3, s2_features_stage3, s1_features_stage3, modalities)
            hr_features_stage3 = vae_out['hr_out']
            s2_features_stage3 = vae_out['s2_out']
            s1_features_stage3 = vae_out['s1_out']
            output['vae_out'] = vae_out
        
        features_stage3 = []
        if 'hr' in self.sources:
            B, C3_G, H3_G, W3_G = hr_features_stage3.shape
            hr_features_stage3 = hr_features_stage3.permute(
                0, 2, 3, 1).reshape(B * H3_G * W3_G, C3_G).unsqueeze(1).contiguous() # B * H3_G * W3_G, 1, C3_G
            features_stage3 = hr_features_stage3

        if 's2' in self.sources:
            # s2_features_stage3 = s2_features[-1]
            _, C3_S2, H3_S2, W3_S2 = s2_features_stage3.shape
            s2_features_stage3 = s2_features_stage3.reshape(
                B, S_S2, C3_S2, H3_S2,
                W3_S2).permute(0, 3, 4, 1, 2).reshape(B, H3_S2 * W3_S2, S_S2,
                                                    C3_S2).contiguous()
            if self.use_ctpe:
                ct_index = sample_list["s2_ct"]
                ctpe = self.ctpe[:, ct_index, :].contiguous().permute(1, 0, 2, 3).contiguous()
                ctpe = ctpe.expand(-1, 256, -1, -1)

                ct_index_2 = sample_list["s2_ct2"]
                ctpe2 = self.ctpe[:, ct_index_2, :].contiguous().permute(1, 0, 2, 3).contiguous()
                ctpe2 = ctpe2.expand(-1, 256, -1, -1)

                ctpe_comb = torch.cat([ctpe, ctpe2], 1)
                # import pdb;pdb.set_trace()
                s2_features_stage3 = (s2_features_stage3 + ctpe_comb).reshape(
                    B * H3_S2 * W3_S2, S_S2, C3_S2).contiguous()
            else:
                s2_features_stage3 = s2_features_stage3.reshape(
                    B * H3_S2 * W3_S2, S_S2, C3_S2).contiguous()
            
            if len(features_stage3) > 0:
                assert H3_G == H3_S2 and W3_G == W3_S2 and C3_G == C3_S2
                features_stage3 = torch.cat((features_stage3, s2_features_stage3), dim=1)
            else:
                features_stage3 = s2_features_stage3

        if 's1' in self.sources:
            # s1_features_stage3 = s1_features[-1]
            _, C3_S1, H3_S1, W3_S1 = s1_features_stage3.shape
            s1_features_stage3 = s1_features_stage3.reshape(
                B, S_S1, C3_S1, H3_S1,
                W3_S1).permute(0, 3, 4, 1, 2).reshape(B, H3_S1 * W3_S1, S_S1,
                                                    C3_S1).contiguous()
            s1_features_stage3 = s1_features_stage3.reshape(
                    B * H3_S1 * W3_S1, S_S1, C3_S1).contiguous()
           
            if len(features_stage3) > 0:
                assert H3_S1 == H3_S2 and W3_S1 == W3_S2 and C3_S1 == C3_S2
                features_stage3 = torch.cat((features_stage3, s1_features_stage3),
                                            dim=1)
            else:
                features_stage3 = s1_features_stage3

        ### 3. fusion
        if self.config.necks.output_cls_token:
            if self.config.necks.get('require_feat', False):
                cls_token, block_outs = self.fusion(features_stage3 , True)
            else:
                cls_token = self.fusion(features_stage3)
                _, C3_G = cls_token.shape
                cls_token = cls_token.reshape(B, H3_G, W3_G,
                                            C3_G).contiguous().permute(0, 3, 1, 2).contiguous() # b, c, h, w
        else:
            assert self.config.necks.with_cls_token is False
            if self.config.necks.get('require_feat', False):
                features_stage3, block_outs = self.fusion(features_stage3, True)
            else:
                features_stage3 = self.fusion(features_stage3)
            features_stage3 = features_stage3.reshape(
                B, H3_S2, W3_S2, S_S2,
                C3_S2).permute(0, 3, 4, 1, 2).reshape(B * S_S2, C3_S2, H3_S2,
                                                   W3_S2).contiguous()
        ### 4. decoder for rec
        hr_rec_inputs = hr_features
        feat_stage1 = hr_rec_inputs[0]

        if feat_stage1.shape[-1] == feat_stage1.shape[-2]:
            feat_stage1_left, feat_stage1_right = torch.split(feat_stage1, feat_stage1.shape[-1] // 2, dim=-1)
            feat_stage1 = torch.cat((feat_stage1_left, feat_stage1_right), dim=1)
            hr_rec_inputs = list(hr_features)
            hr_rec_inputs[0] = feat_stage1

        rec_feats = [*hr_rec_inputs, cls_token]
        logits_hr = self.head_rec_hr(rec_feats)
        if self.config.get('upsacle_results', True):
            logits_hr = logits_hr.to(torch.float32)
            logits_hr = F.interpolate(logits_hr, scale_factor=4, mode='bilinear', align_corners=True)
        output["logits_hr"] = logits_hr
        return output

    def load_pretrained(self, ckpt_path, key):
        pretrained_dict = torch.load(ckpt_path, map_location={'cuda:0': 'cpu'})
        pretrained_dict = pretrained_dict[key]
        for k, v in pretrained_dict.items():
            if k == 'backbone_s2.patch_embed.projection.weight':
                pretrained_in_channels = v.shape[1]
                if self.config.backbone_s2.in_channels == 4:
                    new_weight = v[:, [0, 1, 2, 6]]
                    new_weight = new_weight * (
                        pretrained_in_channels /
                        self.config.backbone_s2.in_channels)
                    pretrained_dict[k] = new_weight
        missing_keys, unexpected_keys = self.load_state_dict(pretrained_dict,
                                                             strict=False)
        print('missing_keys:', missing_keys)
        print('unexpected_keys:', unexpected_keys)

