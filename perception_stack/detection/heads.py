"""Detection heads for 3D object detection."""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """SSD-style detection head for PointPillars.

    Predicts per-anchor:
      - Classification scores (num_classes)
      - Bounding box regression (x, y, z, w, l, h, yaw) = 7
      - Direction classification (2 bins)
    """

    def __init__(self, in_channels: int = 384, num_classes: int = 3, num_anchors: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, 1)
        self.bbox_head = nn.Conv2d(256, num_anchors * 7, 1)
        self.dir_head = nn.Conv2d(256, num_anchors * 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.cls_head, self.bbox_head, self.dir_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        # Bias init for classification (focal loss style)
        nn.init.constant_(self.cls_head.bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Feature map (B, in_channels, H, W)

        Returns:
            Dict with cls_scores (B, num_classes, H, W),
                       bbox_preds (B, 7, H, W),
                       dir_preds (B, 2, H, W)
        """
        shared = self.shared_conv(x)

        cls_scores = self.cls_head(shared)
        bbox_preds = self.bbox_head(shared)
        dir_preds = self.dir_head(shared)

        B, _, H, W = cls_scores.shape
        cls_scores = cls_scores.view(B, self.num_anchors, self.num_classes, H, W)
        cls_scores = cls_scores.max(dim=1)[0]  # (B, num_classes, H, W)

        bbox_preds = bbox_preds.view(B, self.num_anchors, 7, H, W)
        bbox_preds = bbox_preds[:, 0]  # Take first anchor: (B, 7, H, W)

        dir_preds = dir_preds.view(B, self.num_anchors, 2, H, W)
        dir_preds = dir_preds[:, 0]  # (B, 2, H, W)

        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
            "dir_preds": dir_preds,
        }
