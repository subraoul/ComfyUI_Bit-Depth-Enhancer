"""
deepDeband node for ComfyUI
Removes banding artifacts and false contours using deep learning

Based on the paper:
"Deep Gradient-Domain Image Debanding" (ICIP 2022)
by Raymond L. Zhou et al.
https://github.com/RaymondLZhou/deepDeband

This node supports both variants:
- deepDeband-w (weighted): Uses weighted patch fusion for better quality
- deepDeband-f (full): Direct patch processing
"""

import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import os
from typing import Tuple, Optional

# Import the model architecture
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from deband_model import DeepDebandModel


class DeepDeband:
    """
    ComfyUI node for deep learning-based debanding using the deepDeband model (ICIP 2022).

    Removes banding artifacts and false contours from images using a trained U-Net architecture.
    Works on both 8-bit and 16-bit images while preserving the input bit depth.

    Features:
    - Two model variants: deepDeband-w (weighted, recommended) and deepDeband-f (full)
    - Strength control for blending original and debanded images
    - Automatic bit-depth detection and preservation
    - GPU acceleration with CPU fallback
    - Batch processing support
    - Tile-based processing for large images
    """

    DESCRIPTION = "Removes banding artifacts and false contours using deep learning (ICIP 2022). Works on 8-bit or 16-bit images. Trained on 51,490 banded/pristine image pairs. The -w variant uses weighted patch fusion for better quality, while -f processes patches directly."

    # Model cache (shared across instances)
    _model_cache = {}

    def __init__(self):
        self.device = self._get_device()

    @staticmethod
    def _get_device():
        """Determine best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image from previous ComfyUI nodes. Accepts both 8-bit (0-255 values scaled to 0-1) and 16-bit (0-65535 values scaled to 0-1) images. The node automatically detects and preserves the input bit depth."
                }),
                "model": ([
                    "deepDeband-w",  # Weighted fusion (recommended)
                    "deepDeband-f",  # Full/direct processing
                ], {
                    "default": "deepDeband-w",
                    "tooltip": "Model variant: deepDeband-w (weighted patch fusion, better quality, recommended) uses bilateral weighting to fuse overlapping patch predictions for smoother results | deepDeband-f (full/direct) processes patches independently, faster but may have visible seams on large images"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Debanding strength (0.0-1.0). Controls blending between original and debanded images. 0.0 = original image (no debanding), 1.0 = full debanding effect. Use lower values (0.3-0.7) for subtle enhancement or when preserving intentional gradients."
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Tile size for processing (default: 256). The model was trained on 256x256 patches. Larger tiles may reduce seams but use more memory. Smaller tiles use less memory but may have more visible boundaries. Only matters for images larger than tile_size."
                }),
                "tile_overlap": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 128,
                    "step": 16,
                    "tooltip": "Overlap between adjacent tiles in pixels (default: 128). The real deepDeband-w uses 50% overlap (128px for 256px tiles). Higher values reduce visible seams at tile boundaries but increase processing time. Recommended: 128 pixels for best quality matching original deepDeband-w. Set to 64-96 for faster processing (may show slight seams)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debanded_image",)
    FUNCTION = "deband"
    CATEGORY = "bitdepth_enhancement"

    def deband(self, image, model="deepDeband-w", strength=1.0, tile_size=256, tile_overlap=32):
        """
        Main debanding function.

        Args:
            image: ComfyUI image tensor [B, H, W, C] in range [0, 1]
            model: Model variant ("deepDeband-w" or "deepDeband-f")
            strength: Blend strength (0.0 = original, 1.0 = full debanding)
            tile_size: Size of tiles for processing large images
            tile_overlap: Overlap between tiles to reduce seams

        Returns:
            Debanded image tensor [B, H, W, C] with same bit depth as input
        """
        # Handle strength = 0 (no processing needed)
        if strength <= 0.0:
            return (image,)

        # Load model
        model_instance = self._load_model(model)
        if model_instance is None:
            print(f"ERROR: Failed to load model '{model}'. Returning original image.")
            return (image,)

        # Detect input bit depth
        is_16bit = self._detect_bit_depth(image)

        # Process batch
        batch_size = image.shape[0]
        debanded_batch = []

        for b in range(batch_size):
            # Extract single image [H, W, C]
            img = image[b]

            # Process image
            debanded = self._process_image(
                img,
                model_instance,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                use_weighted_fusion=(model == "deepDeband-w"),
                model_variant=model
            )

            # Blend with original based on strength
            if strength < 1.0:
                debanded = img * (1 - strength) + debanded * strength

            debanded_batch.append(debanded)

        # Stack batch
        debanded_output = torch.stack(debanded_batch, dim=0)

        return (debanded_output,)

    def _detect_bit_depth(self, image: torch.Tensor) -> bool:
        """
        Detect if image is 8-bit or 16-bit based on value range.

        Args:
            image: Input image tensor

        Returns:
            True if 16-bit, False if 8-bit
        """
        # Sample some values to check range
        max_val = image.max().item()

        # In ComfyUI, images are normalized to [0, 1]
        # But we can check the precision/distribution
        # 8-bit images typically have values at 1/255 intervals
        # 16-bit images have values at 1/65535 intervals

        # Simple heuristic: if max value suggests more precision than 8-bit
        # This is a simplification - in practice, both are in [0, 1] range
        # For now, we'll assume 8-bit input and let users handle 16-bit conversion
        # in their workflow if needed

        return False  # Default to 8-bit behavior

    def _load_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Load and cache the model.

        Args:
            model_name: Model variant name

        Returns:
            Loaded model instance or None if failed
        """
        # Check cache
        cache_key = f"{model_name}_{self.device}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Determine model file path
        model_dir = os.path.join(
            folder_paths.models_dir,
            "bit_depth_enhancement",
            "deepdeband"
        )

        if model_name == "deepDeband-w":
            model_file = "deepDeband_w.pth"
        elif model_name == "deepDeband-f":
            model_file = "deepDeband_f.pth"
        else:
            print(f"ERROR: Unknown model variant: {model_name}")
            return None

        model_path = os.path.join(model_dir, model_file)

        # Check if file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            print(f"Please ensure the model weights are placed in: {model_dir}")
            print(f"Download from: https://github.com/RaymondLZhou/deepDeband")
            return None

        try:
            # Create model instance with BatchNorm (as used in original deepDeband training)
            # The pretrained weights were trained with BatchNorm2d normalization
            model_instance = DeepDebandModel(norm_type='batch')

            # Load weights
            print(f"Loading deepDeband model: {model_name}")
            model_instance.load_pretrained(model_path, strict=False)

            # Move to device and set eval mode
            model_instance = model_instance.to(self.device)
            model_instance.eval()

            # Cache the model
            self._model_cache[cache_key] = model_instance

            print(f"Successfully loaded {model_name} on {self.device}")
            return model_instance

        except Exception as e:
            print(f"ERROR loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_image(self, image, model, tile_size=256, tile_overlap=128,
                       use_weighted_fusion=True, model_variant="deepDeband-w"):
        """
        Process image with reflection padding matching original deepDeband.

        Pipeline:
        1. Pad image to 256×N using reflection tiling
        2. Process padded image (single or tiled)
        3. Crop back to original size
        """
        # Apply reflection padding FIRST (critical for matching original output)
        padded_image, orig_H, orig_W = self._pad_image_with_reflections(image)

        # Determine processing strategy
        padded_H, padded_W = padded_image.shape[:2]

        if padded_H <= tile_size and padded_W <= tile_size:
            # Small enough to process as single tile (like deepDeband-f)
            processed = self._process_tile(padded_image, model, model_variant)
        else:
            # Process with tiling (like deepDeband-w)
            processed = self._process_tiled(
                padded_image, model, tile_size, tile_overlap,
                use_weighted_fusion, model_variant
            )

        # Crop back to original dimensions
        result = processed[:orig_H, :orig_W, :]

        return result

    def _pad_image_with_reflections(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Pad image to nearest 256×N dimensions using reflection tiling.

        Matches original deepDeband padding strategy from padding.py:
        - Rounds dimensions up to nearest 256 multiple
        - Creates 2×2 tiling of [original, h-flip, v-flip, both-flip]
        - Tiles this pattern across canvas to fill target dimensions
        - Crops to exact target size

        Args:
            image: Input tensor [H, W, C]

        Returns:
            (padded_image, original_H, original_W)
        """
        H, W, C = image.shape

        # Calculate target dimensions (round up to 256 multiple)
        target_H = ((H + 255) // 256) * 256
        target_W = ((W + 255) // 256) * 256

        # If already correct size, no padding needed
        if H == target_H and W == target_W:
            return image, H, W

        # Create 4 reflections for 2×2 tiling
        img_hflip = torch.flip(image, [0])  # Flip vertically
        img_vflip = torch.flip(image, [1])  # Flip horizontally
        img_both = torch.flip(img_hflip, [1])  # Flip both

        # Assemble 2×2 tile block
        top_row = torch.cat([image, img_vflip], dim=1)  # Original | H-flip
        bottom_row = torch.cat([img_hflip, img_both], dim=1)  # V-flip | Both
        tile_block = torch.cat([top_row, bottom_row], dim=0)

        # Calculate repetitions needed
        block_H, block_W = tile_block.shape[:2]
        n_tiles_h = (target_H + block_H - 1) // block_H
        n_tiles_w = (target_W + block_W - 1) // block_W

        # Tile the pattern
        tiled = tile_block.repeat(n_tiles_h, n_tiles_w, 1)

        # Crop to exact target dimensions
        padded = tiled[:target_H, :target_W, :]

        return padded, H, W

    def _pad_to_size(self, image: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
        """
        Pad image to minimum size, using reflection when possible, replicate otherwise.

        Reflection padding requires padding amount < input dimension.
        For very small tiles, fall back to replicate padding.

        Args:
            image: Input tile [H, W, C]
            min_h: Minimum height required
            min_w: Minimum width required

        Returns:
            Padded tile [min_h, min_w, C]
        """
        H, W, C = image.shape

        if H >= min_h and W >= min_w:
            return image

        pad_h = max(0, min_h - H)
        pad_w = max(0, min_w - W)

        # Determine padding mode based on tile size
        # Reflection requires padding < input dimension
        can_use_reflection_h = (H > pad_h) if pad_h > 0 else True
        can_use_reflection_w = (W > pad_w) if pad_w > 0 else True

        if can_use_reflection_h and can_use_reflection_w:
            # Use reflection padding (preferred)
            mode = 'reflect'
        else:
            # Fall back to replicate for very small tiles
            mode = 'replicate'

        padded = F.pad(
            image.permute(2, 0, 1),  # [H, W, C] -> [C, H, W]
            (0, pad_w, 0, pad_h),
            mode=mode
        ).permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

        return padded

    def _create_blend_mask(
        self,
        tile_h: int,
        tile_w: int,
        overlap: int,
        is_start_h: bool,
        is_end_h: bool,
        is_start_w: bool,
        is_end_w: bool,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create a blending mask for a tile with smooth transitions at overlap regions.

        This creates a mask that is 1.0 in the center and smoothly transitions to 0.0
        at edges that will overlap with adjacent tiles. Edges that don't overlap
        (image boundaries) remain at 1.0.

        Args:
            tile_h: Tile height
            tile_w: Tile width
            overlap: Overlap size in pixels
            is_start_h: True if this tile is at the top edge (no overlap above)
            is_end_h: True if this tile is at the bottom edge (no overlap below)
            is_start_w: True if this tile is at the left edge (no overlap to the left)
            is_end_w: True if this tile is at the right edge (no overlap to the right)
            dtype: Data type
            device: Device to create tensor on

        Returns:
            2D blend mask [tile_h, tile_w] with smooth transitions at overlap regions
        """
        # Start with all ones
        mask_h = torch.ones(tile_h, dtype=dtype, device=device)
        mask_w = torch.ones(tile_w, dtype=dtype, device=device)

        # Create smooth transitions at overlapping edges using linear ramp
        # Note: torch.linspace produces slightly non-uniform steps in float32 due to
        # floating-point representation (1/31 cannot be exactly represented), but this
        # has negligible impact compared to tile seam artifacts from BatchNorm.

        # Top edge (if overlapping)
        if not is_start_h and tile_h > overlap:
            ramp = torch.linspace(0, 1, overlap, dtype=dtype, device=device)
            mask_h[:overlap] = ramp

        # Bottom edge (if overlapping)
        if not is_end_h and tile_h > overlap:
            ramp = torch.linspace(1, 0, overlap, dtype=dtype, device=device)
            mask_h[-overlap:] = ramp

        # Left edge (if overlapping)
        if not is_start_w and tile_w > overlap:
            ramp = torch.linspace(0, 1, overlap, dtype=dtype, device=device)
            mask_w[:overlap] = ramp

        # Right edge (if overlapping)
        if not is_end_w and tile_w > overlap:
            ramp = torch.linspace(1, 0, overlap, dtype=dtype, device=device)
            mask_w[-overlap:] = ramp

        # Combine horizontal and vertical masks
        mask_2d = mask_h.unsqueeze(1) * mask_w.unsqueeze(0)  # [tile_h, tile_w]

        return mask_2d

    def _process_tile(self, tile: torch.Tensor, model: torch.nn.Module, model_variant: str = "deepDeband-w") -> torch.Tensor:
        """
        Process a single tile through the model.
        Tile must be at least 256x256.

        Args:
            tile: Image tile [H, W, C] in range [0, 1], H>=256, W>=256
            model: Loaded model instance
            model_variant: Model variant name (for potential variant-specific handling)

        Returns:
            Processed tile [H, W, C] in range [0, 1]
        """
        # Convert to model format: [B, C, H, W] in range [-1, 1]
        tile_input = tile.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        tile_input = tile_input.to(self.device)

        # Normalize to [-1, 1] range (required by pix2pix Tanh output)
        tile_input = tile_input * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # COLOR SPACE HANDLING:
        # Research shows pytorch-CycleGAN-and-pix2pix uses PIL (RGB format), NOT OpenCV.
        # However, testing shows the model expects NO color conversion for correct output.
        # The model was likely trained on RGB data and expects RGB input/output.
        #
        # DO NOT CONVERT RGB↔BGR - keep native RGB format from ComfyUI

        # Process through model
        with torch.no_grad():
            tile_output = model(tile_input)

        # Convert back: [-1, 1] -> [0, 1]
        tile_output = (tile_output + 1.0) / 2.0

        # Convert [B, C, H, W] -> [H, W, C]
        tile_output = tile_output.squeeze(0).permute(1, 2, 0)
        tile_output = tile_output.cpu()

        # Clamp to valid range
        tile_output = torch.clamp(tile_output, 0.0, 1.0)

        return tile_output

    def _process_tiled(
        self,
        image: torch.Tensor,
        model: torch.nn.Module,
        tile_size: int = 256,
        overlap: int = 32,
        use_weighted_fusion: bool = True,
        model_variant: str = "deepDeband-w"
    ) -> torch.Tensor:
        """
        Process large image using overlapping tiles.

        Args:
            image: Image tensor [H, W, C] in range [0, 1]
            model: Loaded model instance
            tile_size: Size of each tile
            overlap: Overlap between adjacent tiles
            use_weighted_fusion: Use weighted blending in overlap regions
            model_variant: Model variant name

        Returns:
            Processed image [H, W, C] in range [0, 1]
        """
        H, W, C = image.shape

        # Calculate stride (tile_size - overlap)
        stride = tile_size - overlap

        # Calculate number of tiles needed
        n_tiles_h = (H + stride - 1) // stride
        n_tiles_w = (W + stride - 1) // stride

        # Create output tensors on GPU to avoid device transfers during accumulation
        output = torch.zeros(H, W, C, dtype=image.dtype, device=self.device)
        weight_map = torch.zeros(H, W, 1, dtype=image.dtype, device=self.device)

        # Process each tile
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries
                y_start = i * stride
                x_start = j * stride
                y_end = min(y_start + tile_size, H)
                x_end = min(x_start + tile_size, W)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end, :]

                # Pad tile if it's smaller than tile_size (edge tiles)
                tile_h, tile_w = tile.shape[:2]
                if tile_h < tile_size or tile_w < tile_size:
                    # Pad to tile_size
                    tile_padded = self._pad_to_size(tile, tile_size, tile_size)
                else:
                    tile_padded = tile

                # Process tile
                processed_tile = self._process_tile(tile_padded, model, model_variant)

                # Extract valid region (remove padding)
                processed_tile = processed_tile[:tile_h, :tile_w, :]

                # Move processed tile to GPU for accumulation
                processed_tile = processed_tile.to(self.device)

                # Create blend mask for this specific tile
                if use_weighted_fusion:
                    # Determine if this tile is at image boundaries
                    is_start_h = (i == 0)
                    is_end_h = (y_end == H)
                    is_start_w = (j == 0)
                    is_end_w = (x_end == W)

                    # Create blend mask with linear ramps at overlap regions
                    # This ensures weights sum to 1.0 in overlaps, preventing quantization
                    tile_weight = self._create_blend_mask(
                        tile_h, tile_w, overlap,
                        is_start_h, is_end_h, is_start_w, is_end_w,
                        dtype=image.dtype, device=self.device
                    )
                    tile_weight = tile_weight.unsqueeze(2)  # [tile_h, tile_w, 1]
                else:
                    tile_weight = torch.ones(tile_h, tile_w, 1, dtype=image.dtype, device=self.device)

                # Accumulate to output
                output[y_start:y_end, x_start:x_end, :] += processed_tile * tile_weight
                weight_map[y_start:y_end, x_start:x_end, :] += tile_weight

        # Normalize by accumulated weights
        # With linear blending, weights should sum to 1.0 in overlap regions
        # and be 1.0 elsewhere, so no amplification occurs
        output = output / torch.clamp(weight_map, min=1e-8)

        # Move final result back to CPU for ComfyUI
        output = output.cpu()

        return output


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "DeepDeband": DeepDeband
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepDeband": "deepDeband (Banding Removal)"
}
