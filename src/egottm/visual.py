"""Adapter for the existing frozen LAM implementation."""

import sys

import torch


def load_lam_model(legacy_root, checkpoint_path, device):
    legacy_root = str(legacy_root)
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    from models.lam.model import LAMBackbone

    model = LAMBackbone(str(checkpoint_path))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def zero_visual_features(batch_size, frames, device, dimension=256):
    return (
        torch.zeros(batch_size, frames, dimension, device=device),
        torch.zeros(batch_size, frames, dtype=torch.bool, device=device),
    )
