"""Semantic segmentation network for lane detection and drivable area."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class LaneSegmentationNet(nn.Module):
    """U-Net style segmentation network for lane detection.

    Outputs:
      - Semantic mask: background / lane_line / drivable_area
      - Lane embedding: per-pixel embeddings for instance clustering
    """

    def __init__(self, num_classes: int = 3, embedding_dim: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(3, 32, stride=2), ResBlock(32))
        self.enc2 = nn.Sequential(ConvBlock(32, 64, stride=2), ResBlock(64))
        self.enc3 = nn.Sequential(ConvBlock(64, 128, stride=2), ResBlock(128))
        self.enc4 = nn.Sequential(ConvBlock(128, 256, stride=2), ResBlock(256))

        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = nn.Sequential(ConvBlock(256, 128), ResBlock(128))

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(128, 64), ResBlock(64))

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(64, 32), ResBlock(32))

        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)

        # Heads
        self.seg_head = nn.Conv2d(32, num_classes, 1)
        self.embed_head = nn.Conv2d(32, embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 3, H, W) input image

        Returns:
            Dict with 'segmentation' (B, num_classes, H, W) and
                       'embedding' (B, embedding_dim, H, W)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections
        d4 = self.up4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)

        seg = self.seg_head(d1)
        embed = self.embed_head(d1)

        return {"segmentation": seg, "embedding": embed}
