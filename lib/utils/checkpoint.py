# Copyright (c) Ant Financial Service Group. and its affiliates.
import os
import warnings

import torch

from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.common.checkpoint import Checkpoint
from antmmf.utils.distributed_utils import is_main_process

class SegCheckpoint(Checkpoint):
    def __init__(self, trainer, load_only=False):
        super().__init__(trainer, load_only=False)

    def load_model_weights(self, file, force=False):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)
        if registry.get(constants.STATE) is constants.STATE_ONLINE_SERVING:
            data_parallel = False
        else:
            data_parallel = registry.get("data_parallel") or registry.get(
                "distributed")

        if "model" in ckpt:
            ckpt_model = ckpt["model"]
        else:
            ckpt_model = ckpt
            ckpt = {"model": ckpt}

        new_dict = {}

        # TODO: Move to separate function
        for attr in ckpt_model:
            if "fa_history" in attr:
                new_dict[attr.replace("fa_history",
                                      "fa_context")] = ckpt_model[attr]
            elif data_parallel is False and attr.startswith("module."):
                new_k = attr.replace("module.", "", 1)
                if '.Wqkv.' in new_k:
                    new_k = new_k.replace('.Wqkv.', '.in_proj_')
                
                new_dict[new_k] = ckpt_model[attr]
            elif data_parallel is not False and not attr.startswith("module."):
                new_dict["module." + attr] = ckpt_model[attr]
            elif data_parallel is False and not attr.startswith("module."):
                print('data_parallel is False and not attr!!!')
                new_k = attr
                if '.Wqkv.' in new_k:
                    new_k = new_k.replace('.Wqkv.', '.in_proj_')
                new_dict[new_k] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]
        print(new_dict.keys())
        self._load_state_dict(new_dict)
        self._load_model_weights_with_mapping(new_dict, force=force)
        print(f'load weight: {file} done!')
        return ckpt

    def _load(self, file, force=False, resume_state=False):
        ckpt = self.load_model_weights(file, force=force)

        # skip loading training state
        if resume_state is False:
            return

        if "optimizer" in ckpt:
            try:
                self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
                # fix the bug of checkpoint in the pytorch with version higher than 1.11
                if "capturable" in self.trainer.optimizer.param_groups[0]:
                    self.trainer.optimizer.param_groups[0]["capturable"] = True
            except Exception as e:
                print(e)
                
        else:
            warnings.warn(
                "'optimizer' key is not present in the checkpoint asked to be loaded. Skipping."
            )

        if "lr_scheduler" in ckpt:
            self.trainer.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        else:
            warnings.warn(
                "'lr_scheduler' key is not present in the checkpoint asked to be loaded. Skipping."
            )

        self.trainer.early_stopping.init_from_checkpoint(ckpt)

        self.trainer.writer.write("Checkpoint {} loaded".format(file))

        if "current_iteration" in ckpt:
            self.trainer.current_iteration = ckpt["current_iteration"]
            registry.register("current_iteration",
                              self.trainer.current_iteration)

        if "current_epoch" in ckpt:
            self.trainer.current_epoch = ckpt["current_epoch"]
            registry.register("current_epoch", self.trainer.current_epoch)

    def save(self, iteration, update_best=False):
        if not is_main_process():
            return

        ckpt_filepath = os.path.join(self.models_foldername,
                                     "model_%d.ckpt" % iteration)
        best_ckpt_filepath = os.path.join(self.ckpt_foldername,
                                          self.ckpt_prefix + "best.ckpt")

        best_iteration = self.trainer.early_stopping.best_monitored_iteration
        best_metric = self.trainer.early_stopping.best_monitored_value
        current_iteration = self.trainer.current_iteration
        current_epoch = self.trainer.current_epoch
        model = self.trainer.model
        data_parallel = registry.get("data_parallel") or registry.get(
            "distributed")

        if data_parallel is True:
            model = model.module

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "current_iteration": current_iteration,
            "current_epoch": current_epoch,
            "best_iteration": best_iteration,
            "best_metric_value": best_metric,
        }

        torch.save(ckpt, ckpt_filepath)
        self.remove_redundant_ckpts()

        if update_best:
            torch.save(ckpt, best_ckpt_filepath)