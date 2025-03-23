# coding: utf-8
# Copyright (c) Ant Group. All rights reserved.
import torch
from torch.distributed import all_reduce, ReduceOp
from antmmf.common.registry import registry
from antmmf.modules.metrics.base_metric import BaseMetric

@registry.register_metric("sem_metric")
class SemMetric(BaseMetric):
    """Segmentation metrics used in evaluation phase.

    Args:
        name (str): Name of the metric.
        eval_type(str): 3 types are supported: 'mIoU', 'mDice', 'mFscore'
        result_field(str): key of predicted results in output dict
        target_field(str): key of ground truth in output dict
        ignore_index(int): class value will be ignored in evaluation
        num_cls(int): total number of categories in evaluation
    """

    def __init__(self,
                 name="dummy_metric", **kwargs
                 ):
        super().__init__(name)
        self.reset()

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Intersection and Union for a batch.

        Args:
            sample_list (Sample_List): data which contains ground truth segmentation maps
            model_output (dict): data which contains prediction segmentation maps
        Returns:
            torch.Tensor: The intersection of prediction and ground truth histogram
                 on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on all
                 classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        return torch.tensor(0).float()

    def reset(self):
        """ initialized all attributes value before evaluation

        """
        self.total_mask_mae = 0
        self.total_num = torch.tensor(0)

    def collect(self, sample_list, model_output, *args, **kwargs):
        """
        Args:
            sample_list(Sample_List): data which contains ground truth segmentation maps
            model_output (Dict): Dict returned by model, that contains two modalities
        Returns:
            torch.FloatTensor: Accuracy
        """
        batch_mask_mae = \
            self.calculate(sample_list, model_output, *args, **kwargs)
        self.total_mask_mae += batch_mask_mae
        self.total_num += 1

    def format(self, *args):
        """ Format evaluated metrics for profile.

        Returns:
            dict: dict of all evaluated metrics.
        """
        output_metric = dict()
        # if self.eval_type == 'mae':
        mae = args[0]
        output_metric['mae'] = mae.item()
        return output_metric

    def summarize(self, *args, **kwargs):
        """This method is used to calculate the overall metric.

        Returns:
            dict: dict of all evaluated metrics.

        """
        # if self.eval_type == 'mae':
        mae = self.total_mask_mae / (self.total_num)
        return self.format(mae)

    def all_reduce(self):
        total_number = torch.stack([
            self.total_mask_mae, self.total_num
        ]).cuda()
        all_reduce(total_number, op=ReduceOp.SUM)
        self.total_mask_mae = total_number[0].cpu()
        self.total_num = total_number[1].cpu()