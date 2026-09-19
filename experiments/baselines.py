"""
Baseline Watermarking Models for Comparative Evaluation:
- HiDDeN (Zhu et al., ECCV 2018)
- StegaStamp (Tancik et al., CVPR 2020)
- SepMark (Wu et al., ACM MM 2023)
- LampMark (Wang et al., AAAI 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiDDeNModel(nn.Module):
    """HiDDeN: Deep network watermarking baseline"""
    def __init__(self, message_length=30):
        super().__init__()
        self.message_length = message_length
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64 + message_length, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
        # Decoder
        self.dec_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dec_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, message_length)
        )

    def encode(self, image, message):
        B, C, H, W = image.shape
        x = F.relu(self.conv1(image))
        msg_exp = message.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        x = torch.cat([x, msg_exp], dim=1)
        x = F.relu(self.conv2(x))
        residual = torch.tanh(self.conv3(x)) * 0.1
        return torch.clamp(image + residual, -1.0, 1.0)

    def decode(self, watermarked_image):
        feat = F.relu(self.dec_conv1(watermarked_image))
        return self.dec_fc(feat)


class StegaStampModel(nn.Module):
    """StegaStamp: Invisible hyperlinks baseline"""
    def __init__(self, message_length=30):
        super().__init__()
        self.message_length = message_length
        # Dense network encoder
        self.fc_msg = nn.Linear(message_length, 32 * 32)
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 3, 3, padding=1)
        # Decoder
        self.dec_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, message_length)
        )

    def encode(self, image, message):
        B, C, H, W = image.shape
        msg_map = self.fc_msg(message).view(B, 1, 32, 32)
        msg_map = F.interpolate(msg_map, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat([image, msg_map], dim=1)
        x = F.relu(self.conv1(x))
        res = torch.tanh(self.conv2(x)) * 0.08
        return torch.clamp(image + res, -1.0, 1.0)

    def decode(self, watermarked_image):
        return self.dec_conv(watermarked_image)


class SepMarkModel(nn.Module):
    """SepMark: Deep separable watermarking baseline"""
    def __init__(self, message_length=30):
        super().__init__()
        self.message_length = message_length
        # Separable Conv Encoder
        self.spatial_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.channel_conv = nn.Conv2d(32, 32, 1)
        self.fc_msg = nn.Linear(message_length, 32)
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)
        # Decoder
        self.dec_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.dec_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, message_length)
        )

    def encode(self, image, message):
        feat = F.relu(self.spatial_conv(image))
        msg_emb = self.fc_msg(message).unsqueeze(-1).unsqueeze(-1)
        feat = feat + msg_emb
        feat = F.relu(self.channel_conv(feat))
        res = torch.tanh(self.out_conv(feat)) * 0.06
        return torch.clamp(image + res, -1.0, 1.0)

    def decode(self, watermarked_image):
        pooled = self.dec_pool(watermarked_image)
        return self.dec_fc(pooled)


class LampMarkModel(nn.Module):
    """LampMark: Localized attention pooling baseline"""
    def __init__(self, message_length=30):
        super().__init__()
        self.message_length = message_length
        # Attention Pooling Encoder & Decoder
        self.encoder_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.dec_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, message_length)
        )

    def encode(self, image, message):
        B, C, H, W = image.shape
        feat = F.relu(self.encoder_conv(image))
        attn_map = self.attn(feat)
        msg_exp = message.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        # Modulate features with attention map and message
        feat = feat + attn_map * msg_exp[:, :64]
        res = torch.tanh(self.out_conv(feat)) * 0.05
        return torch.clamp(image + res, -1.0, 1.0)

    def decode(self, watermarked_image):
        feat = self.dec_conv(watermarked_image)
        return self.dec_fc(feat)
