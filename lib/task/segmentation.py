# coding: utf-8
# Copyright (c) Ant Group. All rights reserved.

from antmmf.common.registry import registry
from antmmf.tasks import BaseTask


@registry.register_task("segmentation")
class SegmentationTask(BaseTask):

    def __init__(self):
        super(SegmentationTask, self).__init__("segmentation")

    def _get_available_datasets(self):
        return ["pretraining_loader"]

    def _preprocess_item(self, item):
        return item
