import torch
import torch.nn.functional as F
from torchvision import transforms

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction


def compute_confidence_map_runtime(img, num_augmentations=4, method='variance'):
    """
    Compute confidence map at runtime using variance over augmentations.
    
    Args:
        img: Input image tensor [C, H, W] with values in [0, 1]
        num_augmentations: Number of augmentations to use for variance computation
        method: 'variance' (variance over augmentations) or 'simple' (simple depth-based)
        
    Returns:
        Confidence map tensor [H, W] with values in [0, 1] (1 = high confidence)
    """
    if method == 'variance':
        # Method: Variance over different augmentations (similar to generate_maps.py var_aug)
        depth_predictions = []
        
        # Original image
        depth_orig = estimate_depth(img, mode='test')
        depth_predictions.append(depth_orig)
        
        # Augmentation 1: Horizontal flip
        img_flipped = torch.flip(img, [2])
        depth_flipped = estimate_depth(img_flipped, mode='test')
        depth_flipped = torch.flip(depth_flipped, [1])  # Flip back
        depth_predictions.append(depth_flipped)
        
        # Augmentation 2: Grayscale (convert to RGB first)
        if num_augmentations > 2:
            img_gray = transforms.Grayscale(num_output_channels=3)(img)
            depth_gray = estimate_depth(img_gray, mode='test')
            depth_predictions.append(depth_gray)
        
        # Augmentation 3: Additive noise
        if num_augmentations > 3:
            img_noise = img + torch.normal(0.0, 0.01, img.size()).to(img.device)
            img_noise = torch.clamp(img_noise, 0.0, 1.0)
            depth_noise = estimate_depth(img_noise, mode='test')
            depth_predictions.append(depth_noise)
        
        # Compute variance across predictions
        depth_stack = torch.stack(depth_predictions, dim=0)  # [N, H, W]
        depth_variance = torch.var(depth_stack, dim=0)  # [H, W]
        
        # Normalize variance to [0, 1]
        if depth_variance.max() > depth_variance.min():
            depth_variance = (depth_variance - depth_variance.min()) / (depth_variance.max() - depth_variance.min() + 1e-8)
        
        # Convert variance (uncertainty) to confidence
        uncertainty = depth_variance
        confidence = 1.0 - uncertainty
        
    elif method == 'simple':
        # Simple method: Use depth gradient as uncertainty proxy
        depth = estimate_depth(img, mode='test')
        
        # Compute depth gradients (higher gradient = more uncertain)
        depth_grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])
        depth_grad_y = torch.abs(depth[1:, :] - depth[:-1, :])
        
        # Pad to match original size
        depth_grad_x = F.pad(depth_grad_x, (0, 1, 0, 0))
        depth_grad_y = F.pad(depth_grad_y, (0, 0, 0, 1))
        
        # Combine gradients
        depth_grad = (depth_grad_x + depth_grad_y) / 2.0
        
        # Normalize and convert to confidence
        if depth_grad.max() > depth_grad.min():
            depth_grad = (depth_grad - depth_grad.min()) / (depth_grad.max() - depth_grad.min() + 1e-8)
        
        confidence = 1.0 - depth_grad
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure confidence is in [0, 1]
    confidence = torch.clamp(confidence, 0.0, 1.0)
    
    return confidence

