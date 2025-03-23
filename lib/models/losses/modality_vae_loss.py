# Copyright (c) Ant Group and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from antmmf.common.registry import registry


@registry.register_loss("ModalityVAELoss")
class ModalityVAELoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.weight = params.pop("weight")
    
    def compute_rec_loss(self, x_in, x_out, modal_flag):
        loss_per_pixel = F.mse_loss(x_in, x_out, reduction='none')
        loss_b = torch.mean(loss_per_pixel, dim=[1, 2, 3])
        return torch.sum(loss_b * modal_flag)/ (modal_flag.sum() + 1e-6)
        
    def forward(self, sample_list, output, *args, **kwargs):
        vae_out = output["vae_out"]
        feat_hr = vae_out['input_hr']
        feat_s2 = vae_out['input_s2']
        feat_s1 = vae_out['input_s1']

        g_hr = vae_out['g_hr']
        g_s2 = vae_out['g_s2'] 
        g_s1 = vae_out['g_s1'] 
        
        # process modality flags
        modality_info = vae_out['modality_info']
        B_M, L_M = modality_info.shape

        modality_hr = modality_info[:,0]
        modality_s2 = modality_info[:,1]
        modality_s1 = modality_info[:,2]

        ######## rec losses ########
        loss_xent = self.compute_rec_loss(g_hr, feat_hr, modality_hr) \
                    + self.compute_rec_loss(g_s2, feat_s2, modality_s2) \
                    + self.compute_rec_loss(g_s1, feat_s1, modality_s1)


        loss_quant = vae_out["loss_quant"]
        total_loss = loss_xent / 3 + loss_quant
        return total_loss * self.weight
