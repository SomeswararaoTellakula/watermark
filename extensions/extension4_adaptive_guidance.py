"""
Extension 4: Adaptive Message-Guided Sampling
Learns optimal guidance scales instead of using fixed values.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from PIL import Image
import numpy as np


class GuidanceScalePredictor(nn.Module):
    """
    Small network that predicts optimal guidance scale per image.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        scale = self.regressor(feat)
        return scale * 4.0  # Output range 0-4


class AdaptiveDiffusionSampler:
    """
    Adaptive diffusion sampler with learned guidance scales.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.guidance_predictor = GuidanceScalePredictor().to(self.device)
        self.optimizer = optim.Adam(self.guidance_predictor.parameters(), lr=1e-4)

    def predict_guidance_scale(self, img: Image.Image) -> float:
        """
        Predict optimal guidance scale for an image.
        Uses content analysis:
        - More guidance for face regions
        - Less guidance for smooth backgrounds
        """
        img_tensor = self._pil_to_tensor(img).to(self.device)

        with torch.no_grad():
            scale = self.guidance_predictor(img_tensor).item()

        # Additional heuristic: check if image contains faces (simplified)
        face_score = self._detect_face_region(img)
        adjusted_scale = scale * (1.0 + face_score * 0.5)

        return min(4.0, adjusted_scale)

    def train_guidance_predictor(
        self,
        images: list,
        optimal_scales: list,
        num_epochs: int = 50,
    ) -> None:
        """
        Train the guidance predictor on labeled examples.
        """
        print(f"🎓 Training adaptive guidance predictor for {num_epochs} epochs...")

        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for img, target_scale in zip(images, optimal_scales):
                self.optimizer.zero_grad()

                img_tensor = self._pil_to_tensor(img).to(self.device)
                pred_scale = self.guidance_predictor(img_tensor)
                target = torch.tensor([target_scale], device=self.device).float()

                loss = criterion(pred_scale.squeeze(), target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(images)
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("✅ Adaptive guidance predictor trained!")

    def adaptive_guided_denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        model: nn.Module,
        guidance_scale: Optional[float] = None,
        img_context: Optional[Image.Image] = None,
    ) -> torch.Tensor:
        """
        Single denoising step with adaptive guidance.
        """
        if guidance_scale is None and img_context is not None:
            guidance_scale = self.predict_guidance_scale(img_context)
        elif guidance_scale is None:
            guidance_scale = 1.0

        # Get model predictions
        with torch.enable_grad():
            x_t.requires_grad_(True)
            eps_pred = model(x_t, torch.tensor([t], device=x_t.device))
            x0_pred = self._predict_x0(x_t, eps_pred, t)

            # Compute guidance
            if guidance_scale > 0:
                loss = self._message_guidance_loss(x0_pred)
                grad = torch.autograd.grad(loss, x_t)[0]
                eps_pred = eps_pred + guidance_scale * grad

        x_prev = self._ddim_step(x_t, eps_pred, t)
        return x_prev

    def _detect_face_region(self, img: Image.Image) -> float:
        """
        Heuristic to detect regions where watermark should be stronger.
        Returns face detection score (0-1).
        """
        arr = np.array(img).astype(np.float32)

        # Simple skin tone detection
        mean_rgb = np.mean(arr, axis=(0, 1))
        r, g, b = mean_rgb / 255.0

        # Skin tone heuristic
        skin_score = 0.0
        if r > g > b and r > 0.3 and r < 0.8:
            skin_score = 0.5 + 0.5 * (r - g)

        # Center bias (faces often in center)
        h, w = arr.shape[:2]
        center_region = arr[
            h//4 : 3*h//4,
            w//4 : 3*w//4
        ]
        center_variance = np.var(center_region)
        if center_variance > 1000:  # High variance often indicates faces
            skin_score += 0.2

        return min(1.0, skin_score)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)

    def _predict_x0(self, x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        """Predict x0 from xt and epsilon."""
        sqrt_alpha_bar = np.cos((t / 100) * np.pi / 2)
        sqrt_one_minus_alpha_bar = np.sin((t / 100) * np.pi / 2)
        return (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar

    def _ddim_step(self, x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        """Simplified DDIM denoising step."""
        return x_t  # Placeholder for actual implementation

    def _message_guidance_loss(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """Message guidance loss function."""
        return torch.sum(x0_pred ** 2)


def demo_adaptive_guidance():
    """Demo function for adaptive guidance."""
    from PIL import Image

    print("Testing Adaptive Message-Guided Sampling...")

    # Create test images
    face_img = Image.new('RGB', (256, 256), color=(200, 150, 150))  # Skin tone
    bg_img = Image.new('RGB', (256, 256), color=(100, 150, 200))    # Background tone

    sampler = AdaptiveDiffusionSampler()

    # Predict guidance scales
    face_scale = sampler.predict_guidance_scale(face_img)
    bg_scale = sampler.predict_guidance_scale(bg_img)

    print(f"✅ Predicted guidance scales:")
    print(f"   - Face image: {face_scale:.2f}")
    print(f"   - Background image: {bg_scale:.2f}")

    # Test training
    train_images = [face_img, bg_img]
    train_scales = [2.5, 0.5]

    sampler.train_guidance_predictor(train_images, train_scales, num_epochs=20)

    # Test after training
    face_scale_after = sampler.predict_guidance_scale(face_img)
    bg_scale_after = sampler.predict_guidance_scale(bg_img)

    print(f"\n✅ After training:")
    print(f"   - Face image: {face_scale_after:.2f} (target: 2.5)")
    print(f"   - Background image: {bg_scale_after:.2f} (target: 0.5)")

    print("\n✅ Adaptive guidance demo complete!")


# Alias for test_all.py compatibility
AdaptiveGuide = AdaptiveDiffusionSampler

def get_guidance_scale(self, img_tensor):
    """Wrapper for test_all.py"""
    # Convert tensor to PIL
    img_arr = ((img_tensor.squeeze().permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    img_pil = Image.fromarray(img_arr)
    return self.predict_guidance_scale(img_pil)

# Add method to class
AdaptiveDiffusionSampler.get_guidance_scale = get_guidance_scale.__get__(AdaptiveDiffusionSampler)

if __name__ == "__main__":
    demo_adaptive_guidance()
