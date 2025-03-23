from .uper_head import UPerHead
from .up_head import UPHead

__all__ = [
    'UPerHead', 'UPHead'
]

type_mapping = {
    'UPerHead': UPerHead,
    'UPHead': UPHead
}


def build_head(type, **kwargs):
    return type_mapping[type](**kwargs)