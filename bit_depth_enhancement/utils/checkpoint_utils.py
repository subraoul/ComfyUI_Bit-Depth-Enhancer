"""
Checkpoint loading utilities for ABCD models

Handles checkpoint format mismatches between saved weights and model architectures
by remapping state dict keys to match expected parameter names.
"""

import torch
from pathlib import Path


def remap_rdn_state_dict(checkpoint_state_dict):
    """
    Remap RDN checkpoint keys to match architecture expectations

    The original ABCD-RDN checkpoint uses a different naming convention:
    - Uses 'convs' instead of 'layers' for dense layer list
    - Has extra '.0' level: 'conv.0.weight' instead of 'conv.weight'
    - Uses 'LFF' instead of 'lff' for local feature fusion

    Transformations:
    1. RDBs.X.convs.Y.conv.0 -> RDBs.X.layers.Y.conv
    2. RDBs.X.LFF -> RDBs.X.lff

    Args:
        checkpoint_state_dict: Original state dict from checkpoint

    Returns:
        Remapped state dict compatible with RDN architecture
    """
    new_state_dict = {}

    for key, value in checkpoint_state_dict.items():
        new_key = key

        # Transform: RDBs.X.convs.Y.conv.0 -> RDBs.X.layers.Y.conv
        if 'RDBs.' in key and '.convs.' in key:
            # Replace 'convs' with 'layers'
            new_key = new_key.replace('.convs.', '.layers.')
            # Remove the extra '.0' after 'conv'
            # Pattern: .conv.0.weight -> .conv.weight
            new_key = new_key.replace('.conv.0.', '.conv.')

        # Transform: LFF -> lff (case change)
        if '.LFF.' in key:
            new_key = new_key.replace('.LFF.', '.lff.')

        new_state_dict[new_key] = value

    return new_state_dict


def remap_swinir_state_dict(checkpoint_state_dict):
    """
    Remap SwinIR checkpoint keys to match architecture expectations

    The original ABCD-SwinIR checkpoint has additional structural levels:
    - Uses 'layers.X.residual_group.blocks.Y' instead of 'layers.X.blocks.Y'
    - Uses 'mlp.fc1' and 'mlp.fc2' instead of 'mlp.0' and 'mlp.2'
    - Contains extra modules (patch_embed, conv_before_upsample) not in our arch
    - Contains 'attn_mask' keys that can be ignored

    Transformations:
    1. layers.X.residual_group.blocks.Y -> layers.X.blocks.Y
    2. mlp.fc1 -> mlp.0, mlp.fc2 -> mlp.2
    3. Skip patch_embed.*, conv_before_upsample.*, and attn_mask keys

    Args:
        checkpoint_state_dict: Original state dict from checkpoint

    Returns:
        Remapped state dict compatible with SwinIR architecture
    """
    new_state_dict = {}

    # Keys to skip (not used in our simplified architecture)
    skip_patterns = ['patch_embed', 'conv_before_upsample', 'attn_mask']

    for key, value in checkpoint_state_dict.items():
        # Skip extra keys that don't exist in our architecture
        # BUT make sure we only skip exact module names, not keys containing them
        should_skip = False
        for pattern in skip_patterns:
            # Match as module name (followed by .)
            if pattern == 'attn_mask':
                # attn_mask is a full key component
                if 'attn_mask' in key:
                    should_skip = True
                    break
            elif f'.{pattern}.' in key or key.startswith(f'{pattern}.') or key.startswith(f'encoder.{pattern}.'):
                should_skip = True
                break

        if should_skip:
            continue

        new_key = key

        # Transform: layers.X.residual_group.blocks.Y -> layers.X.blocks.Y
        # Remove the 'residual_group.' level
        if '.residual_group.blocks.' in key:
            new_key = new_key.replace('.residual_group.blocks.', '.blocks.')

        # Transform: mlp.fc1 -> mlp.0, mlp.fc2 -> mlp.2
        # The checkpoint uses named layers (fc1, fc2) but our Sequential uses indices (0, 2)
        if '.mlp.fc1.' in new_key:
            new_key = new_key.replace('.mlp.fc1.', '.mlp.0.')
        if '.mlp.fc2.' in new_key:
            new_key = new_key.replace('.mlp.fc2.', '.mlp.2.')

        new_state_dict[new_key] = value

    return new_state_dict


def load_abcd_checkpoint(model, checkpoint_path, model_type, strict=True):
    """
    Load ABCD checkpoint with automatic key remapping

    Handles different checkpoint formats and applies model-specific key remapping
    to ensure compatibility between checkpoint and architecture parameter names.

    Args:
        model: ABCD model instance (EDSR/RDN/SwinIR wrapped with ABCD)
        checkpoint_path: Path to checkpoint file (.pth)
        model_type: Model type - 'edsr', 'rdn', or 'swinir'
        strict: If True, raise error on missing/unexpected keys (default: True)

    Returns:
        model: Model with loaded weights

    Raises:
        ValueError: If strict=True and there are missing/unexpected keys
        FileNotFoundError: If checkpoint file doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            if isinstance(checkpoint['model'], dict) and 'sd' in checkpoint['model']:
                state_dict = checkpoint['model']['sd']
            else:
                state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Apply model-specific remapping
    if model_type.lower() == 'rdn':
        print(f"Applying RDN key remapping...")
        original_count = len(state_dict)
        state_dict = remap_rdn_state_dict(state_dict)
        print(f"  Original keys: {original_count}")
        print(f"  Remapped keys: {len(state_dict)}")

    elif model_type.lower() == 'swinir':
        print(f"Applying SwinIR key remapping...")
        original_count = len(state_dict)
        state_dict = remap_swinir_state_dict(state_dict)
        removed_count = original_count - len(state_dict)
        print(f"  Original keys: {original_count}")
        print(f"  Remapped keys: {len(state_dict)}")
        print(f"  Removed keys: {removed_count} (patch_embed, conv_before_upsample)")

    elif model_type.lower() == 'edsr':
        print(f"Loading EDSR checkpoint (no remapping needed)...")
        print(f"  Total keys: {len(state_dict)}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'edsr', 'rdn', or 'swinir'")

    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if strict:
        if missing_keys:
            raise ValueError(
                f"Missing keys in checkpoint for {model_type.upper()}:\n"
                f"  {missing_keys[:10]}\n"
                f"  ... and {len(missing_keys) - 10} more" if len(missing_keys) > 10 else f"  {missing_keys}"
            )
        if unexpected_keys:
            raise ValueError(
                f"Unexpected keys in checkpoint for {model_type.upper()}:\n"
                f"  {unexpected_keys[:10]}\n"
                f"  ... and {len(unexpected_keys) - 10} more" if len(unexpected_keys) > 10 else f"  {unexpected_keys}"
            )

    if missing_keys or unexpected_keys:
        print(f"  Warning: Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"  ✓ All keys loaded successfully!")

    return model


def verify_checkpoint_compatibility(checkpoint_path, model, model_type):
    """
    Verify checkpoint compatibility without loading weights

    Useful for debugging checkpoint format issues.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        model_type: 'edsr', 'rdn', or 'swinir'

    Returns:
        dict with compatibility information
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            if isinstance(checkpoint['model'], dict) and 'sd' in checkpoint['model']:
                state_dict = checkpoint['model']['sd']
            else:
                state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Apply remapping
    if model_type.lower() == 'rdn':
        state_dict = remap_rdn_state_dict(state_dict)
    elif model_type.lower() == 'swinir':
        state_dict = remap_swinir_state_dict(state_dict)

    # Get model keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Compare
    matching = model_keys & checkpoint_keys
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    return {
        'total_model_params': len(model_keys),
        'total_checkpoint_params': len(checkpoint_keys),
        'matching': len(matching),
        'missing_in_checkpoint': sorted(missing),
        'unexpected_in_checkpoint': sorted(unexpected),
        'compatible': len(missing) == 0 and len(unexpected) == 0
    }
