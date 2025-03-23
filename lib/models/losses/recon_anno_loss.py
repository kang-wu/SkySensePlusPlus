# Copyright (c) Ant Group and its affiliates.
import torch
import torch.nn as nn
from antmmf.common.registry import registry
import torch.nn.functional as F

@registry.register_loss("RecLoss")
class RecLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.weight = params.pop("weight")
        self.patch_size = params.pop("patch_size")
        self.eps = torch.finfo(torch.bfloat16).eps
        self.pred_key = params.pop("pred_key")
        self.vocabulary_size = params.pop("vocabulary_size") + 1 
        self.mask_key = params.pop("mask_key")
        self.target_key = params.pop("target_key")
        self.feature_merged = params.pop("feature_merged")
        self.cnt_train = 0
        self.cnt_val = 0
        self.use_bg = params.pop("use_bg")
        if "use_all_patch" in params:
            self.use_all_patch = params.pop("use_all_patch")
        else:
            self.use_all_patch = False
        if "balance" in params:
            self.balance = params.pop("balance")
        else:
            self.balance = False
        if "sim_regularization" in params:
            self.sim_regularization = params.pop("sim_regularization")
        else:
            self.sim_regularization = False

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        w = int((x.shape[1]*0.5)**.5)
        h = w * 2
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p))
        x = torch.einsum('nhwpq->nhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], h * p, w * p))
        return imgs


    def forward(self, sample_list, output, *args, **kwargs):
        pred = output[self.pred_key] # B, C, H, W
        target = output[self.target_key] # B, H, W
        mask = output[self.mask_key]
        b_mask, h_mask, w_mask = mask.shape
        mask = mask.reshape((b_mask, h_mask*w_mask))
        mask = mask[:, :, None].repeat(1, 1, self.patch_size**2)
        mask = self.unpatchify(mask)

        if not self.use_bg:
            valid = sample_list['valid']
            mask = mask * valid

        loss = F.cross_entropy(pred, target, reduction="none")
        
        if self.balance:
            if self.use_all_patch:
                loss_pos = loss[target > 0].sum() / ((target > 0).sum() + 1e-6)
                loss_neg = loss[target == 0].sum() / ((target == 0).sum() + 1e-6)
                loss = (loss_pos + loss_neg) * 0.5
            else:
                loss_pos = loss[(target > 0) & (mask == 1)].sum() / (((target > 0) & (mask == 1)).sum() + 1e-6)
                loss_neg = loss[(target == 0) & (mask == 1)].sum() / (((target == 0) & (mask == 1)).sum() + 1e-6)
                loss = (loss_pos + loss_neg) * 0.5
        else:
            if self.use_all_patch:
                loss = loss.mean()
            else:
                loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if self.sim_regularization:
            vocabulary_token = output['vocabulary_token']
            voca_normed = F.normalize(vocabulary_token, 2, 1)
            similarity_matrix = 1 + torch.einsum('nd,md->nm', voca_normed, voca_normed)
            num = voca_normed.shape[0]
            index = torch.triu(voca_normed.new_ones(num, num), diagonal=1).type(torch.bool)
            loss_reg = similarity_matrix[index].mean()
            return loss * self.weight + loss_reg * 0.05
        return loss * self.weight

