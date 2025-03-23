from antmmf.common.registry import registry
from antmmf.datasets.base_dataset_builder import BaseDatasetBuilder
from .loader.pretraining_loader import PretrainingLoader

@registry.register_builder("pretraining_loader")
class PretrainingBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("pretraining_loader")

    def _build(self, dataset_type, config, *args, **kwargs):
        return None

    def _load(self, dataset_type, config, *args, **kwargs):
        self.dataset = PretrainingLoader(dataset_type, config)
        return self.dataset

    def update_registry_for_model(self, config):
        pass
