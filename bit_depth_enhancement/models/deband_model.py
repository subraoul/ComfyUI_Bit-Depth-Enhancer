"""
DeepDeband Model Architecture
Based on pix2pix UNet Generator for image-to-image translation

This implementation is based on the PyTorch CycleGAN and pix2pix framework
used by the deepDeband paper (ICIP 2022).

The model uses a U-Net architecture with skip connections to preserve
spatial information while learning to remove banding artifacts.

Reference:
- deepDeband: https://github.com/RaymondLZhou/deepDeband
- pix2pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines a single U-Net block with skip connections.

    Structure: Downsampling -> Submodule -> Upsampling -> Concatenation

    The skip connection concatenates input with the processed output,
    preserving spatial information across the network depth.

    Args:
        outer_nc: Number of filters in outer conv layer
        inner_nc: Number of filters in inner conv layer
        input_nc: Number of channels in input (if None, uses outer_nc)
        submodule: Previously defined U-Net submodule (next layer down)
        outermost: Whether this is the outermost (first) layer
        innermost: Whether this is the innermost (bottleneck) layer
        norm_layer: Normalization layer to use (typically nn.BatchNorm2d or nn.InstanceNorm2d)
        use_dropout: Whether to apply dropout in this layer
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        # Downsampling: Conv2d with stride=2
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                            stride=2, padding=1, bias=False if norm_layer != nn.Identity else True)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # Outermost layer: no normalization on downsampling or upsampling
            # Upsampling: ConvTranspose2d with stride=2
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                       kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            # Innermost layer: no skip connection, no normalization on upsampling
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                       kernel_size=4, stride=2, padding=1,
                                       bias=False if norm_layer != nn.Identity else True)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            # Middle layers: include skip connections and dropout if specified
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                       kernel_size=4, stride=2, padding=1,
                                       bias=False if norm_layer != nn.Identity else True)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward pass with skip connection concatenation."""
        if self.outermost:
            return self.model(x)
        else:
            # Add skip connection by concatenating input with output
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """
    U-Net Generator for image-to-image translation.

    This is the core architecture used by deepDeband for removing banding artifacts.
    The U-Net consists of an encoder (downsampling) and decoder (upsampling) with
    skip connections that preserve spatial information.

    Args:
        input_nc: Number of input image channels (3 for RGB)
        output_nc: Number of output image channels (3 for RGB)
        num_downs: Number of downsampling layers in the U-Net (default: 8 for 256x256 images)
        ngf: Number of filters in the first layer (default: 64)
        norm_layer: Normalization layer type (default: BatchNorm2d)
        use_dropout: Whether to use dropout in the inner layers (default: False)

    Architecture:
        - Encoder: Progressive downsampling with increasing channels (ngf -> ngf*8)
        - Bottleneck: Innermost layer with highest feature count
        - Decoder: Progressive upsampling with skip connections from encoder
        - Output: Tanh activation for [-1, 1] range
    """

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # Construct U-Net structure recursively from innermost to outermost
        # Build innermost layer (bottleneck)
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None, innermost=True, norm_layer=norm_layer
        )

        # Add intermediate layers with ngf*8 filters
        # num_downs=8 means we need 6 intermediate layers (8 - 2 for outer/inner)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )

        # Gradually decrease the number of filters in outer layers
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
        )

        # Outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc,
            submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x: Input tensor [B, C, H, W] in range [-1, 1]

        Returns:
            Output tensor [B, C, H, W] in range [-1, 1]
        """
        return self.model(x)


class DeepDebandModel(nn.Module):
    """
    Complete deepDeband model wrapper.

    This class wraps the UnetGenerator and provides a convenient interface
    for loading pretrained weights and running inference.

    The deepDeband model comes in two variants:
    - deepDeband-f (full): Processes entire images or patches directly
    - deepDeband-w (weighted): Uses weighted patch fusion for better results

    Both variants use the same U-Net architecture but differ in how they
    process images (handled at inference time, not in the model itself).

    Usage:
        model = DeepDebandModel()
        model.load_pretrained('/path/to/weights.pth')
        output = model(input_tensor)
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_type='batch'):
        """
        Initialize deepDeband model.

        Args:
            input_nc: Number of input channels (default: 3 for RGB)
            output_nc: Number of output channels (default: 3 for RGB)
            ngf: Number of generator filters in first conv layer (default: 64)
            norm_type: Type of normalization ('batch', 'instance', or 'none')
        """
        super(DeepDebandModel, self).__init__()

        # Select normalization layer
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        # Create U-Net generator
        self.generator = UnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=8,  # Standard for 256x256 patches
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=False  # Typically disabled during inference
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor [B, C, H, W] in range [-1, 1]

        Returns:
            Output tensor [B, C, H, W] in range [-1, 1]
        """
        return self.generator(x)

    def load_pretrained(self, weights_path, strict=True):
        """
        Load pretrained weights from a checkpoint file.

        Args:
            weights_path: Path to .pth weights file
            strict: Whether to strictly enforce that keys match

        Returns:
            Self for method chaining
        """
        # Load checkpoint - may be raw state_dict or wrapped
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Try common wrapper keys
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'generator' in checkpoint:
                state_dict = checkpoint['generator']
            else:
                # Assume it's the raw state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load into generator
        self.generator.load_state_dict(state_dict, strict=strict)
        return self

    def set_eval(self):
        """Set model to evaluation mode."""
        self.eval()
        return self


def create_deepdeband_model(weights_path=None, device='cpu', norm_type='batch'):
    """
    Factory function to create and optionally load a deepDeband model.

    Args:
        weights_path: Optional path to pretrained weights
        device: Device to place model on ('cpu', 'cuda', etc.)
        norm_type: Normalization type ('batch', 'instance', or 'none')

    Returns:
        DeepDebandModel instance, ready for inference

    Example:
        >>> model = create_deepdeband_model('deepDeband_f.pth', device='cuda')
        >>> model.eval()
        >>> output = model(input_tensor)
    """
    model = DeepDebandModel(norm_type=norm_type)

    if weights_path is not None:
        model.load_pretrained(weights_path)
        print(f"Loaded deepDeband weights from: {weights_path}")

    model = model.to(device)
    model.eval()

    return model


# Model dimensions for reference
DEEPDEBAND_INPUT_SIZE = 256  # Model expects 256x256 patches
DEEPDEBAND_INPUT_CHANNELS = 3  # RGB images
DEEPDEBAND_OUTPUT_CHANNELS = 3  # RGB output
