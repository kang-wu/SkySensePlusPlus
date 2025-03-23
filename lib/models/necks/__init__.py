from .transformer_encoder import TransformerEncoder
from .modality_completion import ModalityCompletion

__all__ = ['TransformerEncoder', 'ModalityCompletion']

type_mapping = {
    'TransformerEncoder': TransformerEncoder,
    'ModalityCompletion': ModalityCompletion
}


def build_neck(type, **kwargs):
    return type_mapping[type](**kwargs)