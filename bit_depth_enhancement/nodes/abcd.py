"""
ABCD Bit-Depth Enhancement Node for ComfyUI

ML-based 8-bit to 16-bit enhancement using ABCD (CVPR 2023).
Reconstructs intermediate tonal values for true bit-depth expansion.

Paper: "Learning to Restore Compressed Images with Arbitrary Bit-depth"
Repository: https://github.com/WooKyoungHan/ABCD
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
from pathlib import Path
import math

# Import the ABCD model architectures
import sys
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.abcd_model import make_edsr_abcd, make_rdn_abcd, make_swinir_abcd
from ..utils.checkpoint_utils import load_abcd_checkpoint


class ABCD_BitDepthEnhancement:
    """
    ComfyUI node for ABCD-based bit-depth enhancement (8-bit → 16-bit)

    Uses deep learning models (EDSR, RDN, SwinIR) with coordinate-based
    implicit neural representation for high-quality bit-depth conversion.

    Reduces banding artifacts and reconstructs smooth intermediate tonal values
    for professional cinematography and color grading workflows.
    """

    DESCRIPTION = "ML-based 8-bit to 16-bit enhancement using ABCD (CVPR 2023). Reconstructs intermediate tonal values for true bit-depth expansion."

    # Class-level cache for loaded models
    _model_cache = {}

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input 8-bit image from ComfyUI. Will be enhanced to true 16-bit with learned intermediate tonal values using deep learning."
                }),
                "model": ([
                    "SwinIR-ABCD",  # Default: Best quality, transformer-based
                    "RDN-ABCD",     # Good balance, residual dense network
                    "EDSR-ABCD",    # Fast, enhanced deep residual network
                ], {
                    "default": "SwinIR-ABCD",
                    "tooltip": "ABCD model architecture (all models fully supported with automatic checkpoint remapping): SwinIR-ABCD (Swin Transformer, highest quality, 151M params) | RDN-ABCD (Residual Dense Network, balanced, 132M params) | EDSR-ABCD (Enhanced Deep Residual, fastest, 141M params). All trained on arbitrary bit-depth conversion."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance_bit_depth"
    CATEGORY = "bitdepth_enhancement"

    def get_model_path(self, model_name):
        """
        Get the full path to the model checkpoint

        Args:
            model_name: Model name (e.g., "SwinIR-ABCD")

        Returns:
            Path to model checkpoint file
        """
        # Map model names to filenames
        model_files = {
            "SwinIR-ABCD": "swinir_abcd.pth",
            "RDN-ABCD": "rdn_abcd.pth",
            "EDSR-ABCD": "edsr_abcd.pth"
        }

        filename = model_files[model_name]

        # Try to find model in ComfyUI models directory
        model_dir = Path.home() / "ComfyUI" / "models" / "bit_depth_enhancement" / "abcd"
        model_path = model_dir / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                f"Please ensure model weights are installed at:\n"
                f"  ~/ComfyUI/models/bit_depth_enhancement/abcd/{filename}\n"
                f"\n"
                f"Model weights should have been downloaded during setup."
            )

        return str(model_path)

    def load_model(self, model_name):
        """
        Load or retrieve cached ABCD model

        Args:
            model_name: Model name to load

        Returns:
            Loaded model on appropriate device
        """
        # Check cache first
        cache_key = f"{model_name}_{self.device}"
        if cache_key in self._model_cache:
            print(f"Using cached {model_name} model")
            return self._model_cache[cache_key]

        print(f"Loading {model_name} model...")

        # Get model path
        model_path = self.get_model_path(model_name)

        # Extract model type from model name
        # "EDSR-ABCD" -> 'edsr', "RDN-ABCD" -> 'rdn', "SwinIR-ABCD" -> 'swinir'
        model_type = model_name.replace("-ABCD", "").lower()

        # Create model architecture
        if model_name == "SwinIR-ABCD":
            model = make_swinir_abcd()
        elif model_name == "RDN-ABCD":
            model = make_rdn_abcd()
        elif model_name == "EDSR-ABCD":
            model = make_edsr_abcd()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load checkpoint with automatic key remapping
        try:
            model = load_abcd_checkpoint(model, model_path, model_type, strict=True)
            print(f"Successfully loaded {model_name} from {model_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model checkpoint from {model_path}\n"
                f"Error: {str(e)}\n"
                f"Please ensure the checkpoint file is valid and compatible."
            )

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        # Cache the model
        self._model_cache[cache_key] = model
        print(f"Model cached and ready for inference on {self.device}")

        return model

    def make_coord(self, shape, flatten=True):
        """
        Generate coordinate grid for the image

        Creates normalized coordinates in range [-1, 1] for each pixel position.
        This is used by ABCD's implicit neural representation to query RGB values
        at continuous coordinates.

        Args:
            shape: Image shape (H, W)
            flatten: If True, return flattened coordinates [H*W, 2]

        Returns:
            Coordinate tensor in range [-1, 1]
        """
        h, w = shape

        # Generate coordinate ranges
        # Coordinates are cell-centered: from -1+offset to 1-offset
        y_range = torch.linspace(-1 + 1/h, 1 - 1/h, h)
        x_range = torch.linspace(-1 + 1/w, 1 - 1/w, w)

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')

        # Stack to create coordinate pairs [H, W, 2]
        coord = torch.stack([y_grid, x_grid], dim=-1)

        if flatten:
            coord = coord.view(-1, 2)  # [H*W, 2]

        return coord

    def enhance_bit_depth(self, image, model):
        """
        Main enhancement function using ABCD models

        Args:
            image: ComfyUI image tensor [B, H, W, C] in range [0, 1]
            model: Model architecture name

        Returns:
            Enhanced 16-bit image tensor [B, H, W, C]
        """
        # Load model (uses cache if already loaded)
        abcd_model = self.load_model(model)

        # Clear GPU cache before inference to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get batch size
        batch_size = image.shape[0]
        enhanced_batch = []

        with torch.no_grad():
            for b in range(batch_size):
                # Extract single image [H, W, C]
                img = image[b]

                # Convert to device
                img = img.to(self.device)

                # Convert ComfyUI format [H, W, C] to PyTorch [1, C, H, W]
                img_tensor = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

                H, W = img_tensor.shape[2], img_tensor.shape[3]

                # Generate coordinates for the entire image
                # Coordinates are in range [-1, 1] for grid sampling
                coord = self.make_coord((H, W), flatten=True)  # [H*W, 2]
                coord = coord.unsqueeze(0).to(self.device)  # [1, H*W, 2]

                # Cell size represents the spacing between pixels in normalized coordinates
                # For bit-depth enhancement from 8-bit to 16-bit:
                # basis = 2^(16-8) / (2^16 - 1) ≈ 256/65535 ≈ 0.00390625
                lowbit = 8
                highbit = 16
                basis = (2 ** (highbit - lowbit)) / ((2 ** highbit) - 1)

                # Create cell tensor (constant for all pixels)
                # ABCD uses 1D cell, not 2D
                cell = torch.ones_like(coord[:, :, :1]) * basis  # [1, H*W, 1]
                cell = cell.to(self.device)

                try:
                    # Debug: Print input statistics
                    print(f"\n=== ABCD Inference Debug ({model}) - Batch {b} ===")
                    print(f"Input shape: {img_tensor.shape}")
                    print(f"Input range: [{img_tensor.min():.6f}, {img_tensor.max():.6f}]")
                    print(f"Input mean: {img_tensor.mean():.6f}, std: {img_tensor.std():.6f}")

                    # Generate features from input
                    print(f"Generating features with encoder...")
                    abcd_model.gen_feat(img_tensor)

                    # Debug: Print feature statistics
                    print(f"Feature shape: {abcd_model.feat.shape}")
                    print(f"Feature range: [{abcd_model.feat.min():.6f}, {abcd_model.feat.max():.6f}]")
                    print(f"Coef feature shape: {abcd_model.coef_feat.shape}")
                    print(f"Freq feature shape: {abcd_model.freq_feat.shape}")

                    # Debug: Print coordinate information
                    print(f"Coordinate shape: {coord.shape}")
                    print(f"Coordinate range: [{coord.min():.6f}, {coord.max():.6f}]")
                    print(f"Cell value: {cell[0, 0, 0]:.8f} (basis: {basis:.8f})")
                    print(f"Total query points: {coord.shape[1]} ({H}x{W})")

                    # Query RGB values at coordinates
                    # Output: [1, H*W, 3] in range [0, 1]
                    print(f"Querying RGB at coordinates...")
                    pred = abcd_model.query_rgb(coord, cell)

                    # Debug: Print prediction statistics
                    print(f"Prediction shape: {pred.shape}")
                    print(f"Prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
                    print(f"Prediction mean: {pred.mean():.6f}, std: {pred.std():.6f}")

                    # Reshape to image format [1, H, W, 3]
                    pred = pred.view(1, H, W, 3)

                    # ABCD reconstruction formula: output = pred * basis + input
                    # This combines the predicted high-frequency details with the input
                    img_for_residual = img_tensor.permute(0, 2, 3, 1)  # [1, H, W, C]
                    enhanced = pred * basis + img_for_residual

                    # Debug: Print enhancement statistics
                    print(f"Enhanced (before clamp) range: [{enhanced.min():.6f}, {enhanced.max():.6f}]")
                    print(f"Difference from input (mean abs): {(enhanced - img_for_residual).abs().mean():.6f}")

                    # Clamp to valid range [0, 1]
                    enhanced = torch.clamp(enhanced, 0.0, 1.0)

                    # Debug: Print final statistics
                    print(f"Enhanced (after clamp) range: [{enhanced.min():.6f}, {enhanced.max():.6f}]")
                    print(f"Enhanced mean: {enhanced.mean():.6f}, std: {enhanced.std():.6f}")
                    print(f"=== Inference Complete ===\n")

                    # Remove batch dimension [H, W, C]
                    enhanced = enhanced.squeeze(0)

                    # Convert back to CPU for ComfyUI
                    enhanced_batch.append(enhanced.cpu())

                except RuntimeError as e:
                    # Handle GPU memory errors specifically
                    if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                        print(f"GPU memory error for batch {b}: {str(e)}")
                        print(f"Attempting CPU fallback for image size {H}x{W}")

                        # Clear GPU cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        try:
                            # Retry on CPU
                            cpu_device = torch.device("cpu")
                            abcd_model_cpu = abcd_model.to(cpu_device)
                            img_tensor_cpu = img_tensor.to(cpu_device)
                            coord_cpu = coord.to(cpu_device)
                            cell_cpu = cell.to(cpu_device)

                            # Generate features and query on CPU
                            abcd_model_cpu.gen_feat(img_tensor_cpu)
                            pred = abcd_model_cpu.query_rgb(coord_cpu, cell_cpu)
                            pred = pred.view(1, H, W, 3)

                            img_for_residual = img_tensor_cpu.permute(0, 2, 3, 1)
                            enhanced = pred * basis + img_for_residual
                            enhanced = torch.clamp(enhanced, 0.0, 1.0)
                            enhanced = enhanced.squeeze(0)

                            enhanced_batch.append(enhanced)

                            # Move model back to original device
                            abcd_model.to(self.device)

                            print(f"CPU fallback successful for batch {b}")
                        except Exception as cpu_error:
                            print(f"CPU fallback also failed: {str(cpu_error)}")
                            # Return original image as last resort
                            enhanced_batch.append(img.cpu())
                    else:
                        print(f"Error during inference for batch {b}: {str(e)}")
                        # On error, return original image
                        enhanced_batch.append(img.cpu())

                except Exception as e:
                    print(f"Unexpected error during inference for batch {b}: {str(e)}")
                    # On error, return original image
                    enhanced_batch.append(img.cpu())

                # Clear GPU cache after each image to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Stack batch
        enhanced_output = torch.stack(enhanced_batch, dim=0)

        return (enhanced_output,)


# ====================================================================
# Node Registration
# ====================================================================

NODE_CLASS_MAPPINGS = {
    "ABCD_BitDepthEnhancement": ABCD_BitDepthEnhancement
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ABCD_BitDepthEnhancement": "ABCD Bit-Depth Enhancement (8→16)"
}
