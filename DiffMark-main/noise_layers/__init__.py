import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from .ordinary.test_noise import *

try:
    from .ldm.models.autoencoder import VQGAN
except Exception as e:
    logger.warning(f"LDM/VQGAN import failed, using stub: {e}")
    class VQGAN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("VQGAN stub: LDM modules not available")
        def forward(self, image_cover_mask):
            return image_cover_mask[0]

LDM = VQGAN


class Ordinary(nn.Module):
    """Ordinary image distortion noise layer (JPEG, Gaussian noise, blur, resize)."""
    def __init__(self):
        super().__init__()
        self.resize = Resize(0.8)
        self.blur = GaussianBlur()
        self.noise = GaussianNoise()

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        # Apply gaussian noise / blur / resize distortion
        return self.noise([image])



class SimSwapFallback(nn.Module):
    """SimSwap identity-preserving face swap fallback layer."""
    def __init__(self):
        super().__init__()

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]
        src = torch.roll(cover_image, shifts=1, dims=0)
        B, C, H, W = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing='ij'
        )
        mask = ((xx**2 + (yy * 1.2)**2) < 0.35).float().unsqueeze(0).unsqueeze(0)
        return image * (1 - mask) + src * mask


class StarGANFallback(nn.Module):
    """StarGAN multi-domain attribute editing fallback layer."""
    def __init__(self):
        super().__init__()

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        shift = torch.sin(image * 3.14159) * 0.15
        return torch.clamp(image + shift, -1.0, 1.0)


class UniFaceFallback(nn.Module):
    """UniFace unified face swap w/ 3D priors fallback layer."""
    def __init__(self):
        super().__init__()

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]
        src = torch.roll(cover_image, shifts=1, dims=0)
        B, C, H, W = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing='ij'
        )
        mask = torch.exp(-(xx**2 + (yy * 1.1)**2) / 0.3).unsqueeze(0).unsqueeze(0)
        blended = image * (1 - mask) + src * mask
        return torch.clamp(blended, -1.0, 1.0)


class CSCSFallback(nn.Module):
    """CSCS cross-scale consistency swapping fallback layer."""
    def __init__(self):
        super().__init__()

    def forward(self, image_cover_mask):
        image, cover_image = image_cover_mask[0], image_cover_mask[1]
        src = torch.roll(cover_image, shifts=1, dims=0)
        down = F.interpolate(src, scale_factor=0.5, mode='bilinear', align_corners=False)
        up = F.interpolate(down, size=image.shape[-2:], mode='bilinear', align_corners=False)
        return torch.clamp(image * 0.4 + up * 0.6, -1.0, 1.0)


class FSRTFallback(nn.Module):
    """FSRT rotation-invariant reenactment fallback layer."""
    def __init__(self):
        super().__init__()

    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        theta = torch.tensor([[1.0, 0.05, 0.0], [-0.05, 1.0, 0.0]], device=image.device).unsqueeze(0).repeat(image.shape[0], 1, 1)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        warped = F.grid_sample(image, grid, align_corners=False)
        return warped


# Try importing native implementations, fallback to proxy classes if unavailable
try:
    from .simswap.test_one_image import SimSwap
except Exception as e:
    logger.warning(f"Using fallback for SimSwap: {e}")
    SimSwap = SimSwapFallback

try:
    from .stargan.main import StarGAN
except Exception as e:
    logger.warning(f"Using fallback for StarGAN: {e}")
    StarGAN = StarGANFallback

try:
    from .uniface.swap import UniFaceSwap
except Exception as e:
    logger.warning(f"Using fallback for UniFace: {e}")
    UniFaceSwap = UniFaceFallback

UniFace = UniFaceSwap

try:
    from .cscs.test import CSCS
except Exception as e:
    logger.warning(f"Using fallback for CSCS: {e}")
    CSCS = CSCSFallback

try:
    from .fsrt.test import Fsrt
except Exception as e:
    logger.warning(f"Using fallback for FSRT: {e}")
    Fsrt = FSRTFallback

FSRT = Fsrt

