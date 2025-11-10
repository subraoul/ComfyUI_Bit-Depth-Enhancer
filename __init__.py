"""
ComfyUI Bit Depth Enhancement
Classical and deep learning-based bit-depth enhancement for professional cinematography workflows

This is the ComfyUI entry point that loads the custom nodes.
"""

from .bit_depth_enhancement.nodes.classical import BitDepthEnhancementClassical, Save16BitTIFF
from .bit_depth_enhancement.nodes.abcd import ABCD_BitDepthEnhancement
from .bit_depth_enhancement.nodes.deband import DeepDeband

NODE_CLASS_MAPPINGS = {
    "BitDepthEnhancementClassical": BitDepthEnhancementClassical,
    "Save16BitTIFF": Save16BitTIFF,
    "ABCD_BitDepthEnhancement": ABCD_BitDepthEnhancement,
    "DeepDeband": DeepDeband,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BitDepthEnhancementClassical": "Bit Depth Enhancement (Classical)",
    "Save16BitTIFF": "Save 16-bit TIFF",
    "ABCD_BitDepthEnhancement": "ABCD Bit-Depth Enhancement (8→16)",
    "DeepDeband": "deepDeband (Banding Removal)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
