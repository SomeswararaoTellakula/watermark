"""
Extension 10: Defense Against Watermark Removal Attacks
Makes watermarks resistant to dedicated removal attacks:
- Adversarial training against removal networks
- Robust invisible watermarking
- Active detection of removal attempts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from PIL import Image
import numpy as np


class WatermarkRemovalNetwork(nn.Module):
    """
    Simulated watermark removal network for adversarial training.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.decoder(feat)


class RobustWatermarkEncoder(nn.Module):
    """
    Watermark encoder trained adversarially against removal networks.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.watermark_injector = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, watermark_bits: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.encoder(x)
        if watermark_bits is not None:
            # Inject watermark
            bits_reshaped = watermark_bits.view(-1, 1, 1, 1).repeat(1, feat.size(1), feat.size(2), feat.size(3))
            feat = feat + bits_reshaped * 0.1
        feat = self.watermark_injector(feat)
        out = self.decoder(feat)
        return x + 0.02 * out


class RemovalDetector(nn.Module):
    """
    Detects if a watermark removal attempt has been made.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


class AdversarialWatermarkTrainer:
    """
    Adversarial training pipeline for robust watermarking.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.watermark_encoder = RobustWatermarkEncoder().to(self.device)
        self.removal_network = WatermarkRemovalNetwork().to(self.device)
        self.detector = RemovalDetector().to(self.device)

        self.encoder_optimizer = optim.Adam(self.watermark_encoder.parameters(), lr=1e-4)
        self.removal_optimizer = optim.Adam(self.removal_network.parameters(), lr=1e-4)
        self.detector_optimizer = optim.Adam(self.detector.parameters(), lr=1e-4)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def train_step(
        self,
        img_batch: torch.Tensor,
        watermark_bits: torch.Tensor,
    ) -> dict:
        """
        Single adversarial training step.
        """
        batch_size = img_batch.size(0)

        # Phase 1: Train removal network
        self.removal_optimizer.zero_grad()

        watermarked = self.watermark_encoder(img_batch, watermark_bits)
        removed = self.removal_network(watermarked)

        removal_loss = self.mse_loss(removed, img_batch)
        removal_loss.backward()
        self.removal_optimizer.step()

        # Phase 2: Train watermark encoder to resist removal
        self.encoder_optimizer.zero_grad()

        watermarked = self.watermark_encoder(img_batch, watermark_bits)
        removed = self.removal_network(watermarked).detach()

        # Reconstruction loss
        recon_loss = self.mse_loss(watermarked, img_batch)

        # Robustness loss: watermark should survive removal attempt
        # We want the difference between watermarked and removed to be small
        robustness_loss = self.mse_loss(removed, img_batch)

        # Combined loss
        encoder_loss = recon_loss - 0.3 * robustness_loss
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # Phase 3: Train detector
        self.detector_optimizer.zero_grad()

        pos_labels = torch.ones(batch_size, 1, device=self.device)
        neg_labels = torch.zeros(batch_size, 1, device=self.device)

        pos_pred = self.detector(watermarked.detach())
        neg_pred = self.detector(img_batch)

        pos_loss = self.bce_loss(pos_pred, pos_labels)
        neg_loss = self.bce_loss(neg_pred, neg_labels)
        detector_loss = pos_loss + neg_loss

        detector_loss.backward()
        self.detector_optimizer.step()

        return {
            "removal_loss": removal_loss.item(),
            "encoder_loss": encoder_loss.item(),
            "detector_loss": detector_loss.item(),
        }

    def train(self, num_steps: int = 100, batch_size: int = 4) -> None:
        """
        Full adversarial training.
        """
        print(f"🛡️ Starting adversarial training for {num_steps} steps...")

        for step in range(num_steps):
            # Create dummy batch
            img_batch = torch.randn(batch_size, 3, 256, 256).to(self.device)
            watermark_bits = torch.randint(0, 2, (batch_size, 30)).float().to(self.device)

            metrics = self.train_step(img_batch, watermark_bits)

            if (step + 1) % 20 == 0:
                print(f"   Step {step+1}/{num_steps}")
                print(f"      Removal loss: {metrics['removal_loss']:.4f}")
                print(f"      Encoder loss: {metrics['encoder_loss']:.4f}")
                print(f"      Detector loss: {metrics['detector_loss']:.4f}")

        print("\n✅ Adversarial training complete!")


class RobustWatermarker:
    """
    Complete robust watermarker with removal defense.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = AdversarialWatermarkTrainer(device)

    def embed_robust(self, img: Image.Image, message: str) -> Tuple[Image.Image, list]:
        """Embed robust watermark."""
        bits = []
        for c in message:
            bits.extend([(ord(c) >> i) & 1 for i in range(8)])
        bits = bits[:30] + [0] * (30 - len(bits[:30]))

        img_tensor = self._pil_to_tensor(img).to(self.device)
        bits_tensor = torch.tensor(bits).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            wm_tensor = self.trainer.watermark_encoder(img_tensor, bits_tensor)
            wm_tensor = torch.clamp(wm_tensor, -1, 1)

        return self._tensor_to_pil(wm_tensor), bits

    def detect_removal(self, img: Image.Image) -> Tuple[bool, float]:
        """Detect if watermark removal has been attempted."""
        img_tensor = self._pil_to_tensor(img).to(self.device)
        with torch.no_grad():
            score = self.trainer.detector(img_tensor).item()
        return score > 0.5, score

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        arr = tensor.squeeze(0).cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = (np.clip(arr, -1, 1) + 1.0) * 127.5
        return Image.fromarray(arr.astype(np.uint8))


def demo_removal_defense():
    """Demo function for removal defense."""
    print("Testing Watermark Removal Defense...")

    watermarker = RobustWatermarker()

    # Test adversarial training
    print("\n1. Starting adversarial training...")
    watermarker.trainer.train(num_steps=50, batch_size=2)

    # Test embedding and detection
    print("\n2. Testing watermarking and removal detection...")
    test_img = Image.new('RGB', (256, 256), color='lightyellow')

    wm_img, bits = watermarker.embed_robust(test_img, "SecretMessage")
    print("   Watermark embedded successfully")

    # Simulate removal
    import random
    arr = np.array(wm_img).astype(np.float32)
    arr += np.random.normal(0, 5, arr.shape)
    removed_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Detect removal
    is_removed, score = watermarker.detect_removal(removed_img)
    print(f"   Removal detected: {is_removed} (score: {score:.4f})")

    print("\n✅ Watermark removal defense demo complete!")


# Alias for test_all.py compatibility
RemovalDefense = RobustWatermarker

if __name__ == "__main__":
    demo_removal_defense()
