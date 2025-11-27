#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import os
import cv2
import numpy as np


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l1_loss_confidence(network_output, gt, confidence=None):
    """
    L1 loss with confidence weighting.
    
    Args:
        network_output: Predicted image tensor [C, H, W] or [B, C, H, W]
        gt: Ground truth image tensor [C, H, W] or [B, C, H, W]
        confidence: Confidence map tensor [H, W] or [B, H, W] or [B, 1, H, W]
                   Values should be in [0, 1] where 1 = high confidence
                   
    Returns:
        Weighted L1 loss (scalar)
    """
    pixel_loss = torch.abs(network_output - gt)  # [C, H, W] or [B, C, H, W]
    
    if confidence is None:
        return pixel_loss.mean()
    
    # Handle different confidence shapes
    if len(confidence.shape) == 2:  # [H, W]
        confidence = confidence.unsqueeze(0)  # [1, H, W]
    if len(confidence.shape) == 3 and confidence.shape[0] != pixel_loss.shape[0]:  # [H, W] -> [1, H, W]
        confidence = confidence.unsqueeze(0)
    if len(confidence.shape) == 3 and confidence.shape[1] != 1:  # [B, H, W] -> [B, 1, H, W]
        confidence = confidence.unsqueeze(1)
    
    # Ensure confidence has same spatial dimensions
    if confidence.shape[-2:] != pixel_loss.shape[-2:]:
        confidence = F.interpolate(confidence, size=pixel_loss.shape[-2:], mode='bilinear', align_corners=False)
    
    # Expand confidence to match network_output channels if needed
    if len(pixel_loss.shape) == 4:  # [B, C, H, W]
        if confidence.shape[1] == 1:
            confidence = confidence.expand(-1, pixel_loss.shape[1], -1, -1)
    elif len(pixel_loss.shape) == 3:  # [C, H, W]
        if len(confidence.shape) == 4:  # [B, 1, H, W]
            confidence = confidence[0, 0]  # [H, W]
        if confidence.shape[0] == 1:
            confidence = confidence.expand(pixel_loss.shape[0], -1, -1)  # [C, H, W]
    
    # Weighted loss: (pixel_loss * confidence).mean() / confidence.mean()
    # Normalize by mean confidence to maintain scale
    weighted_loss = (pixel_loss * confidence).sum() / (confidence.sum() + 1e-8)
    return weighted_loss

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l2_loss_confidence(network_output, gt, confidence=None):
    """
    L2 loss with confidence weighting.
    
    Args:
        network_output: Predicted image tensor [C, H, W] or [B, C, H, W]
        gt: Ground truth image tensor [C, H, W] or [B, C, H, W]
        confidence: Confidence map tensor [H, W] or [B, H, W] or [B, 1, H, W]
                   Values should be in [0, 1] where 1 = high confidence
                   
    Returns:
        Weighted L2 loss (scalar)
    """
    pixel_loss = ((network_output - gt) ** 2)  # [C, H, W] or [B, C, H, W]
    
    if confidence is None:
        return pixel_loss.mean()
    
    # Handle different confidence shapes (same as l1_loss_confidence)
    if len(confidence.shape) == 2:  # [H, W]
        confidence = confidence.unsqueeze(0)  # [1, H, W]
    if len(confidence.shape) == 3 and confidence.shape[0] != pixel_loss.shape[0]:  # [H, W] -> [1, H, W]
        confidence = confidence.unsqueeze(0)
    if len(confidence.shape) == 3 and confidence.shape[1] != 1:  # [B, H, W] -> [B, 1, H, W]
        confidence = confidence.unsqueeze(1)
    
    # Ensure confidence has same spatial dimensions
    if confidence.shape[-2:] != pixel_loss.shape[-2:]:
        confidence = F.interpolate(confidence, size=pixel_loss.shape[-2:], mode='bilinear', align_corners=False)
    
    # Expand confidence to match network_output channels if needed
    if len(pixel_loss.shape) == 4:  # [B, C, H, W]
        if confidence.shape[1] == 1:
            confidence = confidence.expand(-1, pixel_loss.shape[1], -1, -1)
    elif len(pixel_loss.shape) == 3:  # [C, H, W]
        if len(confidence.shape) == 4:  # [B, 1, H, W]
            confidence = confidence[0, 0]  # [H, W]
        if confidence.shape[0] == 1:
            confidence = confidence.expand(pixel_loss.shape[0], -1, -1)  # [C, H, W]
    
    # Weighted loss
    weighted_loss = (pixel_loss * confidence).sum() / (confidence.sum() + 1e-8)
    return weighted_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim_confidence(img1, img2, confidence=None, window_size=11, size_average=True):
    """
    SSIM loss with confidence weighting.
    
    Args:
        img1: First image tensor [C, H, W] or [B, C, H, W]
        img2: Second image tensor [C, H, W] or [B, C, H, W]
        confidence: Confidence map tensor [H, W] or [B, H, W] or [B, 1, H, W]
                   Values should be in [0, 1] where 1 = high confidence
        window_size: Size of the SSIM window
        size_average: Whether to average over spatial dimensions
        
    Returns:
        Weighted SSIM value (scalar)
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Compute SSIM map
    ssim_map = _ssim(img1, img2, window, window_size, channel, size_average=False)
    
    if confidence is None:
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    # Handle different confidence shapes
    if len(confidence.shape) == 2:  # [H, W]
        confidence = confidence.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif len(confidence.shape) == 3:  # [B, H, W] or [H, W, 1]
        if confidence.shape[0] == img1.shape[0] if len(img1.shape) == 4 else 1:
            confidence = confidence.unsqueeze(1)  # [B, 1, H, W]
        else:
            confidence = confidence.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Ensure confidence has same spatial dimensions as SSIM map
    if len(ssim_map.shape) == 4:  # [B, C, H, W]
        if confidence.shape[-2:] != ssim_map.shape[-2:]:
            confidence = F.interpolate(confidence, size=ssim_map.shape[-2:], mode='bilinear', align_corners=False)
        if confidence.shape[1] == 1:
            confidence = confidence.expand(-1, ssim_map.shape[1], -1, -1)
    elif len(ssim_map.shape) == 3:  # [C, H, W]
        if len(confidence.shape) == 4:
            confidence = confidence[0]  # [1, H, W]
        if confidence.shape[-2:] != ssim_map.shape[-2:]:
            confidence = F.interpolate(confidence.unsqueeze(0), size=ssim_map.shape[-2:], mode='bilinear', align_corners=False)[0]
        if confidence.shape[0] == 1:
            confidence = confidence.expand(ssim_map.shape[0], -1, -1)
    
    # Weighted SSIM: (ssim_map * confidence).mean() / confidence.mean()
    weighted_ssim = (ssim_map * confidence).sum() / (confidence.sum() + 1e-8)
    
    return weighted_ssim

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def load_confidence_map(confidence_path, normalize=True):
    """
    Load confidence map from file (PNG or numpy array).
    
    Args:
        confidence_path: Path to confidence map file
        normalize: If True, normalize to [0, 1] range
        
    Returns:
        Confidence map as torch.Tensor [H, W] with values in [0, 1]
    """
    if not os.path.exists(confidence_path):
        raise FileNotFoundError(f"Confidence map not found: {confidence_path}")
    
    # Load as image (16-bit PNG from generate_maps.py)
    if confidence_path.endswith('.png') or confidence_path.endswith('.jpg'):
        confidence_img = cv2.imread(confidence_path, cv2.IMREAD_UNCHANGED)
        if confidence_img is None:
            raise ValueError(f"Could not load image: {confidence_path}")
        
        # Convert to float and normalize
        if confidence_img.dtype == np.uint16:
            confidence = confidence_img.astype(np.float32) / 65535.0
        elif confidence_img.dtype == np.uint8:
            confidence = confidence_img.astype(np.float32) / 255.0
        else:
            confidence = confidence_img.astype(np.float32)
        
        # If grayscale, take first channel
        if len(confidence.shape) == 3:
            confidence = confidence[:, :, 0]
    else:
        # Load as numpy array
        confidence = np.load(confidence_path)
        if isinstance(confidence, np.ndarray):
            confidence = confidence.astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {confidence_path}")
    
    # Normalize to [0, 1] if needed
    if normalize:
        if confidence.max() > 1.0 or confidence.min() < 0.0:
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
    
    # Convert to torch tensor
    confidence_tensor = torch.from_numpy(confidence).float()
    
    return confidence_tensor


def uncertainty_to_confidence(uncertainty_map, method='inverse'):
    """
    Convert uncertainty map to confidence map.
    
    Args:
        uncertainty_map: Uncertainty map tensor [H, W] with values in [0, 1]
        method: Conversion method
            - 'inverse': confidence = 1 - uncertainty
            - 'exp': confidence = exp(-uncertainty)
            - 'normalized': confidence = (1 - uncertainty) / (1 + uncertainty)
            
    Returns:
        Confidence map tensor [H, W] with values in [0, 1]
    """
    if method == 'inverse':
        confidence = 1.0 - uncertainty_map
    elif method == 'exp':
        confidence = torch.exp(-uncertainty_map)
        # Normalize to [0, 1]
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
    elif method == 'normalized':
        confidence = (1.0 - uncertainty_map) / (1.0 + uncertainty_map + 1e-8)
        # Normalize to [0, 1]
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return confidence.clamp(0.0, 1.0)

