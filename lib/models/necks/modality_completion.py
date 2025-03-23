# Copyright (c) AntGroup. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class BFloat16UpsampleNearest2d(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x_float = x.float() 
        upsampled_x = F.interpolate(x_float, scale_factor=self.scale_factor, mode=self.mode)
        return upsampled_x.to(x.dtype)

class ConvVQVAEv2(nn.Module):
    def __init__(self, input_shape, conv_dim, z_dim, num_tokens=8192, temp=0.9):
        super().__init__()
        self.z_dim = z_dim
        self.conv_dim = conv_dim  # 256
        self.input_shape = input_shape  # 256
        self.temp = temp
        # code book
        self.codebook = nn.Embedding(num_tokens, z_dim)
        # encoder
        self.relu = nn.LeakyReLU()
        self.pool = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(input_shape[0], conv_dim, 5, stride=1, padding=2)
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.gamma_1 = nn.Parameter(0.001 * torch.ones((1, conv_dim, 1, 1)))
        self.enc_block2 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.gamma_2 = nn.Parameter(0.001 * torch.ones((1, conv_dim, 1, 1)))
        self.logit_conv = nn.Conv2d(conv_dim, num_tokens, 1)
        # decoder
        self.unpool = BFloat16UpsampleNearest2d(scale_factor=2)
        self.conv2 = nn.Conv2d(z_dim, conv_dim, 3, stride=1, padding=1)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.gamma_3 = nn.Parameter(0.001 * torch.ones((1, conv_dim, 1, 1)))
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.gamma_4 = nn.Parameter(0.001 * torch.ones((1, conv_dim, 1, 1)))
        self.rec_conv = nn.Conv2d(conv_dim, input_shape[0], 3, stride=1, padding=1)

    def forward_encoder(self, x):
        x = self.relu(self.conv1(x))
        x = x + self.gamma_1 * self.enc_block1(x)
        x = self.pool(x)
        x = x + self.gamma_2 * self.enc_block2(x)
        x = self.pool(x)
        logits = self.logit_conv(x)
        return logits

    def forward_decoder(self, logits):
        soft_one_hot = F.softmax(logits * (self.temp*10), dim=1)
        sampled = torch.einsum('bnhw,nd->bdhw', soft_one_hot, self.codebook.weight)
        x = self.relu(self.conv2(sampled))
        x = self.unpool(x)
        x = x + self.gamma_3 * self.dec_block1(x)
        x = self.unpool(x)
        x = x + self.gamma_4 * self.dec_block2(x)
        rec_feats = self.rec_conv(x)
        return rec_feats, soft_one_hot

    def forward(self, x):
        print(x.shape)
        logits = self.forward_encoder(x)
        images_p, soft_one_hot = self.forward_decoder(logits)
        return [logits, images_p]

class ModalityCompletion(nn.Module):
    def __init__(self,
                 input_shape_hr=(2816, 16, 16),
                 input_shape_s2=(2816, 16, 16),
                 input_shape_s1=(2816, 16, 16),
                 conv_dim=256,
                 z_dim=256,
                 n_codebook=8192,
                 init_cfg=None
                ):
        super(ModalityCompletion, self).__init__()
        self.vae_hr = ConvVQVAEv2(input_shape=input_shape_hr, conv_dim=conv_dim, z_dim=z_dim, num_tokens=n_codebook)
        self.vae_s2 = ConvVQVAEv2(input_shape=input_shape_s2, conv_dim=conv_dim, z_dim=z_dim, num_tokens=n_codebook)
        self.vae_s1 = ConvVQVAEv2(input_shape=input_shape_s1, conv_dim=conv_dim, z_dim=z_dim, num_tokens=n_codebook)
        self.kl_div_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.init_cfg=init_cfg

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            from mmcls.utils import get_root_logger
            from mmcv.runner import CheckpointLoader, load_state_dict
            logger = get_root_logger()
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            load_state_dict(self, state_dict, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def kl_loss(self, logits_hr, logits_s2, logits_s1, modality_info):
        prob_hr = F.log_softmax(logits_hr, dim=1)
        prob_s2 = F.log_softmax(logits_s2, dim=1)
        prob_s1 = F.log_softmax(logits_s1, dim=1)
        flag_hr = modality_info[:,0][:, None, None, None]
        flag_s2 = modality_info[:,1][:, None, None, None]
        flag_s1 = modality_info[:,2][:, None, None, None]
        loss_hr_s2 = self.kl_div_loss(prob_hr, prob_s2) + self.kl_div_loss(prob_s2, prob_hr)
        loss_hr_s2 = (loss_hr_s2 * flag_hr * flag_s2).sum((1, 2, 3)).mean()
        loss_hr_s1 = self.kl_div_loss(prob_hr, prob_s1) + self.kl_div_loss(prob_s1, prob_hr)
        loss_hr_s1 = (loss_hr_s1 * flag_hr * flag_s1).sum((1, 2, 3)).mean()
        loss_s2_s1 = self.kl_div_loss(prob_s2, prob_s1) + self.kl_div_loss(prob_s1, prob_s2)
        loss_s2_s1 = (loss_s2_s1 * flag_s2 * flag_s1).sum((1, 2, 3)).mean()
        loss = (loss_hr_s2 + loss_hr_s1 + loss_s2_s1) / 6.0

        return loss

    def forward(self, feat_hr, feat_s2, feat_s1, modality_info):
        # encoders，add noise
        # each modality
        # 2816, 16, 16 => conv 256, 4, 4 => flatten 4096(256*4*4) => linear mu 256, log_var 256
        B, C, H, W = feat_hr.shape
        B_M, L_M = modality_info.shape
        assert B == B_M, f'feat_hr batch: {B}, modality_info batch: {B_M}'

        # quant, emb_loss, info
        # hr input flow
        logits_hr = self.vae_hr.forward_encoder(feat_hr)
        logits_s2 = self.vae_s2.forward_encoder(feat_s2)
        logits_s1 = self.vae_s1.forward_encoder(feat_s1)
        modality_hr = modality_info[:,0]
        modality_s2 = modality_info[:,1]
        modality_s1 = modality_info[:,2]
        flag_hr = modality_hr[:, None, None, None] # B => B, C, H, W
        flag_s2 = modality_s2[:, None, None, None]
        flag_s1 = modality_s1[:, None, None, None]

        mean_logits_hr_s2 = logits_hr * flag_hr + logits_s2 * flag_s2
        mean_logits_hr_s1 = logits_hr * flag_hr + logits_s1 * flag_s1
        mean_logits_s1_s2 = logits_s1 * flag_s1 + logits_s2 * flag_s2

        logits_hr_rec  = logits_hr * flag_hr + mean_logits_s1_s2 * (~flag_hr)
        logits_s2_rec  = logits_s2 * flag_s2 + mean_logits_hr_s1 * (~flag_s2)
        logits_s1_rec  = logits_s1 * flag_s1 + mean_logits_hr_s2 * (~flag_s1)
        g_hr, soft_one_hot_hr =  self.vae_hr.forward_decoder(logits_hr_rec)
        g_s2, soft_one_s2 =  self.vae_s2.forward_decoder(logits_s2_rec)
        g_s1, soft_one_s1 =  self.vae_s1.forward_decoder(logits_s1_rec)

        hr_out = feat_hr * flag_hr + g_hr * (~flag_hr)
        s2_out = feat_s2 * flag_s2 + g_s2 * (~flag_s2)
        s1_out = feat_s1 * flag_s1 + g_s1 * (~flag_s1)

        output = {}

        output['hr_out'] = hr_out
        output['s2_out'] = s2_out
        output['s1_out'] = s1_out

        output['modality_info'] = modality_info

        output['input_hr'] = feat_hr
        output['input_s2'] = feat_s2
        output['input_s1'] = feat_s1

        output['logits_hr'] = logits_hr
        output['logits_s2'] = logits_s2
        output['logits_s1'] = logits_s1

        output['soft_one_hot_hr'] = soft_one_hot_hr
        output['soft_one_hot_s2'] = soft_one_s2
        output['soft_one_hot_s1'] = soft_one_s1

        output['g_hr'] = g_hr
        output['g_s2'] = g_s2
        output['g_s1'] = g_s1
        output['loss_quant'] = self.kl_loss(logits_hr, logits_s2, logits_s1, modality_info)

        return output

