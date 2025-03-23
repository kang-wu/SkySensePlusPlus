import numpy as np


def cosine_scheduler(base_value,
                     final_value,
                     all_iters,
                     warmup_iters=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(all_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == all_iters
    return schedule


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def cancel_gradients_backbone(iteration, model, freeze_backbone_steps):
    if iteration >= freeze_backbone_steps:
        return
    for n, p in model.named_parameters():
        if "backbon_hr" in n or 'backbon_s2' in n or 'head_s2' in n or 'fusion' in n or 'ctpe' in n:
            p.grad = None


class EMA():

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay
                               ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class LayerDecayValueAssigner(object):

    def __init__(self, layer_decay, num_layers, base_lr, net_type, arch='huge'):
        assert net_type in ['swin', 'vit']
        assert 0 < layer_decay <= 1
        depths_dict = {
            'tiny': [2, 2, 6, 2],
            'small': [2, 2, 18, 2],
            'base': [2, 2, 18, 2],
            'large': [2, 2, 18, 2],
            'huge': [2, 2, 18, 2],
            'giant': [2, 2, 42, 4],
        }
        num_layers = num_layers if net_type == 'vit' else sum(depths_dict[arch])
        self.layer_decay = layer_decay
        self.values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        self.depths = depths_dict[arch]
        self.base_lr = base_lr
        self.net_type = net_type

    def get_num_layer_for_vit(self, var_name):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("layers"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return len(self.values) - 1

    def get_num_layer_for_swin(self, var_name):
        if var_name in ("mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("stages"):
            layer_id = int(var_name.split('.')[1])
            if 'blocks' in var_name:
                block_id = int(var_name.split('.')[3])
            else:
                block_id = self.depths[layer_id] - 1
            layer_id = sum(self.depths[:layer_id]) + block_id
            return layer_id + 1
        else:
            return len(self.values) - 1

    def get_layer_id(self, var_name):
        if self.net_type == 'swin':
            return self.get_num_layer_for_swin(var_name)
        if self.net_type == 'vit':
            return self.get_num_layer_for_vit(var_name)

    def fix_param(self, model, num_block=4):
        if num_block < 1:
            return 0
        frozen_num = 0
        if self.net_type == 'swin':
            for name, param in model.named_parameters():
                if name.startswith("patch_embed"):
                    param.requires_grad = False
                    frozen_num += 1
                if name.startswith("stages") and self.get_layer_id(name) <= num_block:
                    param.requires_grad = False
                    frozen_num += 1
        if self.net_type == 'vit':
            for name, param in model.named_parameters():
                if name.startswith("patch_embed"):
                    param.requires_grad = False
                    frozen_num += 1
                if name.startswith("layers") and self.get_layer_id(name) <= num_block:
                    param.requires_grad = False
                    frozen_num += 1
        return frozen_num
    
    def fix_param_deeper(self, model, num_block=4):
        if num_block < 1:
            return 0
        frozen_num = 0
        if self.net_type == 'swin':
            raise ValueError('Not Support')
        if self.net_type == 'vit':
            for name, param in model.named_parameters():
                if name.startswith("patch_embed"):
                    param.requires_grad = False
                    frozen_num += 1
                if name.startswith("layers") and self.get_layer_id(name) >= num_block:
                    param.requires_grad = False
                    frozen_num += 1
        return frozen_num

    def get_parameter_groups(self, model, weight_decay):
        parameter_groups_with_wd, parameter_groups_without_wd = [], []
        print_info_with_wd, print_info_without_wd = [], []
        no_decay = [
            "absolute_pos_embed", "relative_position_bias_table", "norm", "bias"
        ]
        if self.layer_decay == 1:
            parameter_groups_with_wd.append(
                {"params": [], "weight_decay": weight_decay, "lr": self.base_lr}
            )
            print_info_with_wd.append(
                {"params": [], "weight_decay": weight_decay, "lr": self.base_lr}
            )
            parameter_groups_without_wd.append(
                {"params": [], "weight_decay": 0, "lr": self.base_lr}
            )
            print_info_without_wd.append(
                {"params": [], "weight_decay": 0, "lr": self.base_lr}
            )
        else:
            for scale in self.values:
                parameter_groups_with_wd.append(
                    {"params": [], "weight_decay": weight_decay, "lr": scale * self.base_lr}
                )
                print_info_with_wd.append(
                    {"params": [], "weight_decay": weight_decay, "lr": scale * self.base_lr}
                )
                parameter_groups_without_wd.append(
                    {"params": [], "weight_decay": 0, "lr": scale * self.base_lr}
                )
                print_info_without_wd.append(
                    {"params": [], "weight_decay": 0, "lr": scale * self.base_lr}
                )
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f'frozen param: {name}')
                continue  # frozen weights
            layer_id = self.get_layer_id(name) if self.layer_decay < 1 else 0
            if any(nd in name for nd in no_decay):
                parameter_groups_without_wd[layer_id]['params'].append(param)
                print_info_without_wd[layer_id]['params'].append(name)
            else:
                parameter_groups_with_wd[layer_id]['params'].append(param)
                print_info_with_wd[layer_id]['params'].append(name)
        parameter_groups_with_wd = [x for x in parameter_groups_with_wd if len(x['params']) > 0]
        parameter_groups_without_wd = [x for x in parameter_groups_without_wd if len(x['params']) > 0]
        print_info_with_wd = [x for x in print_info_with_wd if len(x['params']) > 0]
        print_info_without_wd = [x for x in print_info_without_wd if len(x['params']) > 0]
        if self.layer_decay < 1:
            for wd, nwd in zip(print_info_with_wd, print_info_without_wd):
                print(wd)
                print(nwd)
        parameter_groups = []
        parameter_groups.extend(parameter_groups_with_wd)
        parameter_groups.extend(parameter_groups_without_wd)
        return parameter_groups

