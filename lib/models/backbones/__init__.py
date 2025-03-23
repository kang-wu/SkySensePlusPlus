from .swin_v2 import SwinTransformerV2MSL
from .vit import VisionTransformerMSL

__all__ = [
    'SwinTransformerV2MSL', 'VisionTransformerMSL'
]

type_mapping = {
    'SwinTransformerV2MSL': SwinTransformerV2MSL,
    'VisionTransformerMSL': VisionTransformerMSL
}

def build_backbone(type, **kwargs):
    return type_mapping[type](**kwargs)
