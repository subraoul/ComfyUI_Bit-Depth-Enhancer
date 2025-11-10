"""
ABCD (Arbitrary Bit-depth Converter and Dequantizer) Model Architectures

This module implements the three ABCD model variants for bit-depth enhancement:
- EDSR_ABCD: Enhanced Deep Residual Networks with ABCD wrapper
- RDN_ABCD: Residual Dense Networks with ABCD wrapper
- SwinIR_ABCD: Swin Transformer with ABCD wrapper

Based on: https://github.com/WooKyoungHan/ABCD
Paper: "Learning to Restore Compressed Images with Arbitrary Bit-depth"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================
# Base Components (shared across all architectures)
# ====================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron used as the implicit function network (imnet)

    Maps concatenated features (coefficient * frequency + cell) to RGB values.
    """
    def __init__(self, in_dim, out_dim, hidden_list):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension (typically 3 for RGB)
            hidden_list: List of hidden layer dimensions
        """
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP_DirectRGB(nn.Module):
    """
    MLP variant that outputs RGB directly from the last hidden layer

    Used for RDN and SwinIR ABCD checkpoints where layer 6 outputs [3, 256]
    instead of [256, 256]. This is the correct architecture for the ABCD paper.
    """
    def __init__(self, in_dim, out_dim, hidden_list):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension (3 for RGB)
            hidden_list: List of hidden layer dimensions (should be 4 layers of 256)
        """
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        # Final layer outputs RGB directly (3 channels) from last hidden dimension
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ====================================================================
# EDSR Components
# ====================================================================

class ResBlock(nn.Module):
    """
    Residual block for EDSR

    Architecture: Conv -> ReLU -> Conv -> Residual connection (scaled)
    """
    def __init__(self, n_feats, kernel_size=3, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        padding = kernel_size // 2

        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=padding)
        )

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class MeanShift(nn.Conv2d):
    """
    Mean shift layer for RGB normalization

    Subtracts/adds RGB mean values using a 1x1 convolution
    """
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class EDSR(nn.Module):
    """
    Enhanced Deep Residual Networks (EDSR) encoder

    Original EDSR paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution"
    This version is configured for ABCD (no upsampling, feature extraction only)
    """
    def __init__(self, n_resblocks=38, n_feats=128, n_colors=3, res_scale=1.0, no_upsampling=True):
        """
        Args:
            n_resblocks: Number of residual blocks (default 38 for EDSR-baseline)
            n_feats: Number of feature channels (default 128)
            n_colors: Number of input/output color channels (default 3 for RGB)
            res_scale: Residual scaling factor
            no_upsampling: If True, skip upsampling (used for ABCD)
        """
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2

        # RGB mean normalization
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        # Head: initial feature extraction (as Sequential to match checkpoint)
        self.head = nn.Sequential(
            nn.Conv2d(n_colors, n_feats, kernel_size, padding=padding)
        )

        # Body: residual blocks
        body = [ResBlock(n_feats, kernel_size, res_scale) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=padding))
        self.body = nn.Sequential(*body)

        self.no_upsampling = no_upsampling

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor [B, n_feats, H, W]
        """
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        return res


# ====================================================================
# RDN Components
# ====================================================================

class DenseLayer(nn.Module):
    """Dense layer for RDN - Conv + ReLU with concatenation"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat([x, out], 1)


class RDB(nn.Module):
    """
    Residual Dense Block (RDB)

    Multiple dense layers with local feature fusion
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

        # Local Feature Fusion
        self.lff = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, 1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.lff(out)
        # Local residual learning
        return out + x


class RDN(nn.Module):
    """
    Residual Dense Network (RDN) encoder

    Original RDN paper: "Residual Dense Network for Image Super-Resolution"
    This version is configured for ABCD (no upsampling, feature extraction only)
    """
    def __init__(self, n_colors=3, G0=64, RDNkSize=3, RDNconfig='B', no_upsampling=True):
        """
        Args:
            n_colors: Number of input color channels
            G0: Number of feature channels
            RDNkSize: Kernel size for RDN layers
            RDNconfig: Configuration preset ('A', 'B', or 'C')
            no_upsampling: If True, skip upsampling (used for ABCD)
        """
        super().__init__()

        # Configuration presets
        configs = {
            'A': {'D': 20, 'C': 6, 'G': 32},  # D: num RDBs, C: layers per RDB, G: growth rate
            'B': {'D': 16, 'C': 8, 'G': 64},
            'C': {'D': 8, 'C': 8, 'G': 64}
        }
        config = configs[RDNconfig]

        self.D = config['D']
        self.C = config['C']
        self.G = config['G']
        self.G0 = G0

        # Shallow feature extraction
        self.SFENet1 = nn.Conv2d(n_colors, G0, RDNkSize, padding=RDNkSize//2)
        self.SFENet2 = nn.Conv2d(G0, G0, RDNkSize, padding=RDNkSize//2)

        # Residual dense blocks
        self.RDBs = nn.ModuleList([RDB(G0, self.G, self.C) for _ in range(self.D)])

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            nn.Conv2d(self.D * G0, G0, 1),
            nn.Conv2d(G0, G0, RDNkSize, padding=RDNkSize//2)
        )

        self.no_upsampling = no_upsampling

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor [B, G0, H, W]
        """
        f_1 = self.SFENet1(x)
        x = self.SFENet2(f_1)

        RDBs_out = []
        for rdb in self.RDBs:
            x = rdb(x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f_1  # Global residual learning

        return x


# ====================================================================
# SwinIR Components
# ====================================================================

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self Attention (W-MSA) for Swin Transformer
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Window partition
        x_windows = self.window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)"""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size, mlp_ratio)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)

        # Reshape to image format for conv
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class SwinIR(nn.Module):
    """
    SwinIR: Image Restoration Using Swin Transformer

    Original SwinIR paper: "SwinIR: Image Restoration Using Swin Transformer"
    This version is configured for ABCD (no upsampling, feature extraction only)
    """
    def __init__(self, img_size=48, in_chans=3, embed_dim=180, depths=[6, 6, 6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6, 6, 6], window_size=8, mlp_ratio=2., no_upsampling=True):
        """
        Args:
            img_size: Input image size
            in_chans: Number of input channels
            embed_dim: Patch embedding dimension
            depths: Depth of each Swin Transformer block
            num_heads: Number of attention heads in each layer
            window_size: Window size for attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            no_upsampling: If True, skip upsampling (used for ABCD)
        """
        super().__init__()

        self.window_size = window_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.no_upsampling = no_upsampling

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor [B, embed_dim, H, W]
        """
        B, C, H, W = x.shape

        # Check input size
        H_padded = ((H + self.window_size - 1) // self.window_size) * self.window_size
        W_padded = ((W + self.window_size - 1) // self.window_size) * self.window_size

        if H != H_padded or W != W_padded:
            x = F.pad(x, (0, W_padded - W, 0, H_padded - H), mode='reflect')

        # Shallow feature extraction
        x_first = self.conv_first(x)

        # Deep feature extraction
        x = x_first.flatten(2).transpose(1, 2)  # B, HW, C
        for layer in self.layers:
            x = layer(x, (H_padded, W_padded))

        x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.embed_dim, H_padded, W_padded)
        x = self.conv_after_body(x) + x_first

        # Remove padding if added
        if H != H_padded or W != W_padded:
            x = x[:, :, :H, :W]

        return x


# ====================================================================
# ABCD Wrapper Models
# ====================================================================

class ABCD(nn.Module):
    """
    ABCD (Arbitrary Bit-depth Converter and Dequantizer) wrapper

    Wraps encoder networks (EDSR/RDN/SwinIR) with coordinate-based
    implicit neural representation for arbitrary bit-depth conversion.

    Architecture:
    1. Encoder extracts features from input image
    2. Coefficient and frequency branches process features
    3. Implicit function (MLP) predicts RGB values at continuous coordinates
    4. Output is combined with input for residual learning
    """
    def __init__(self, encoder, hidden_dim=256, imnet_hidden=[256, 256, 256, 256], num_in_ch=None, use_direct_rgb=False):
        """
        Args:
            encoder: Backbone encoder (EDSR/RDN/SwinIR)
            hidden_dim: Hidden dimension for coefficient and frequency branches
            imnet_hidden: Hidden layer dimensions for implicit function
            num_in_ch: Number of input channels from encoder (if None, auto-detect)
            use_direct_rgb: If True, use MLP_DirectRGB (for RDN/SwinIR checkpoints)
        """
        super().__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim

        # Get encoder output channels
        if num_in_ch is not None:
            encoder_dim = num_in_ch
        elif hasattr(encoder, 'G0'):
            encoder_dim = encoder.G0  # RDN
        elif hasattr(encoder, 'embed_dim'):
            encoder_dim = encoder.embed_dim  # SwinIR
        else:
            encoder_dim = 128  # EDSR default

        # Coefficient and frequency branches
        self.coef = nn.Conv2d(encoder_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(encoder_dim, hidden_dim, 3, padding=1)

        # Implicit function network (coordinate-based)
        # Input: hidden_dim (coef * freq) + 1 (cell size)
        # Note: Original ABCD uses 1D cell, not 2D
        # RDN and SwinIR checkpoints use direct RGB output architecture
        if use_direct_rgb:
            self.imnet = MLP_DirectRGB(hidden_dim + 1, 3, imnet_hidden)
        else:
            self.imnet = MLP(hidden_dim + 1, 3, imnet_hidden)

        # Output activation (normalized to [0, 1])
        self.tanh = nn.Tanh()

    def gen_feat(self, inp):
        """
        Generate features from input image

        Args:
            inp: Input tensor [B, C, H, W]
        """
        self.feat = self.encoder(inp)
        self.coef_feat = self.coef(self.feat)
        self.freq_feat = self.freq(self.feat)

    def query_rgb(self, coord, cell):
        """
        Query RGB values at continuous coordinates

        Args:
            coord: Coordinates [B, N, 2] in range [-1, 1]
            cell: Cell sizes [B, N, 1] (scalar cell size)

        Returns:
            RGB values [B, N, 3]
        """
        # Sample coefficient and frequency features at coordinates
        # coord needs to be flipped for grid_sample (y, x) -> (x, y)
        coord_flip = coord.flip(-1)

        # grid_sample expects [B, H, W, 2]
        B, N, _ = coord.shape
        coord_sample = coord_flip.view(B, 1, -1, 2)

        coef = F.grid_sample(
            self.coef_feat, coord_sample,
            mode='bilinear', align_corners=False
        )  # [B, hidden_dim, 1, N]

        freq = F.grid_sample(
            self.freq_feat, coord_sample,
            mode='bilinear', align_corners=False
        )  # [B, hidden_dim, 1, N]

        coef = coef.view(B, self.hidden_dim, N).permute(0, 2, 1)  # [B, N, hidden_dim]
        freq = freq.view(B, self.hidden_dim, N).permute(0, 2, 1)  # [B, N, hidden_dim]

        # Frequency modulation: coef * (cos(2π * freq) + 1) / 2
        # This creates a smooth interpolation function
        freq_encoded = torch.cos(math.pi * freq) * 0.5 + 0.5
        modulated = coef * freq_encoded

        # Concatenate with cell information (1D cell size)
        inp = torch.cat([modulated, cell], dim=-1)

        # Predict RGB residual
        ret = self.imnet(inp)

        # Normalize to [0, 1] using scaled tanh
        ret = (self.tanh(ret) + 1) / 2

        return ret

    def forward(self, inp, coord, cell):
        """
        Forward pass

        Args:
            inp: Input tensor [B, C, H, W]
            coord: Coordinates [B, N, 2]
            cell: Cell sizes [B, N, 2]

        Returns:
            RGB values [B, N, 3]
        """
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


# ====================================================================
# Factory Functions for Creating ABCD Models
# ====================================================================

def make_edsr_abcd(hidden_dim=256, imnet_hidden=[256, 256, 256, 256]):
    """
    Create EDSR-ABCD model

    Configuration: EDSR-baseline with 38 residual blocks, 128 features
    Note: EDSR uses standard MLP architecture (not direct RGB)
    """
    encoder = EDSR(n_resblocks=38, n_feats=128, res_scale=1.0, no_upsampling=True)
    model = ABCD(encoder, hidden_dim, imnet_hidden, use_direct_rgb=False)
    return model


def make_rdn_abcd(hidden_dim=256, imnet_hidden=[256, 256, 256]):
    """
    Create RDN-ABCD model

    Configuration: RDN config 'C' with 8 RDBs, 8 layers per RDB, growth rate 64
    Note: Using config 'C' (8 blocks) to match ABCD checkpoint architecture
    Note: RDN checkpoint uses direct RGB output (layer 6: [3, 256])
    """
    encoder = RDN(G0=64, RDNconfig='C', no_upsampling=True)
    model = ABCD(encoder, hidden_dim, imnet_hidden, use_direct_rgb=True)
    return model


def make_swinir_abcd(hidden_dim=256, imnet_hidden=[256, 256, 256], img_size=64,
                     embed_dim=180, depths=[6, 6, 6, 6, 6, 6],
                     num_heads=[6, 6, 6, 6, 6, 6], window_size=8, mlp_ratio=2):
    """
    Create SwinIR-ABCD model

    Configuration: SwinIR-Light variant with 6 RSTB blocks
    Note: Checkpoint uses embed_dim=180, num_heads=6 (not the default 64/8)
    Note: SwinIR checkpoint uses direct RGB output (layer 6: [3, 256])
    Note: Encoder has embed_dim=180 internally, but outputs 64 channels to ABCD wrapper
    """
    encoder = SwinIR(
        img_size=img_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        no_upsampling=True
    )
    model = ABCD(encoder, hidden_dim, imnet_hidden, num_in_ch=64, use_direct_rgb=True)
    return model
