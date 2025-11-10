"""
ComfyUI Bit Depth Enhancement
Classical and deep learning-based bit-depth enhancement for professional cinematography workflows
"""

__version__ = "0.1.0"

from .nodes.classical import BitDepthEnhancementClassical, Save16BitTIFF
from .nodes.deband import DeepDeband

NODE_CLASS_MAPPINGS = {
    "BitDepthEnhancementClassical": BitDepthEnhancementClassical,
    "Save16BitTIFF": Save16BitTIFF,
    "DeepDeband": DeepDeband,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BitDepthEnhancementClassical": "Bit Depth Enhancement (Classical)",
    "Save16BitTIFF": "Save 16-bit TIFF",
    "DeepDeband": "deepDeband (Banding Removal)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
