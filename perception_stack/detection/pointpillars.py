"""PointPillars-style 3D object detection.

Reference: Lang et al., 'PointPillars: Fast Encoders for Object Detection from Point Clouds' (CVPR 2019).

Architecture:
  1. Pillar Feature Net — groups points into vertical pillars, encodes with PointNet-like MLP
  2. Backbone — 2D CNN over pseudo-image of pillar features
  3. Detection Head — predicts class, bbox regression, and direction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from perception_stack.detection.heads import DetectionHead
from perception_stack.detection.nms import nms_3d


class PillarFeatureNet(nn.Module):
    """Encodes raw point cloud pillars into fixed-size feature vectors.

    Each point is augmented with (xc, yc, zc, xp, yp) where c=centroid offset, p=pillar center.
    A shared MLP + max-pool produces one feature vector per pillar.
    """

    def __init__(self, in_channels: int = 4, pillar_feat_channels: int = 64,
                 max_points_per_pillar: int = 32):
        super().__init__()
        # Input: (x, y, z, intensity) + 5 augmented = 9 channels
        augmented_channels = in_channels + 5
        self.max_points = max_points_per_pillar

        self.linear = nn.Linear(augmented_channels, pillar_feat_channels, bias=False)
        self.bn = nn.BatchNorm1d(pillar_feat_channels)

    def forward(
        self,
        pillar_points: torch.Tensor,
        pillar_coords: torch.Tensor,
        num_points_per_pillar: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pillars.

        Args:
            pillar_points: (N_pillars, max_points, C) padded point data
            pillar_coords: (N_pillars, 2) grid coordinates (x_idx, y_idx)
            num_points_per_pillar: (N_pillars,) actual point count per pillar

        Returns:
            pillar_features: (N_pillars, D) encoded features
        """
        N, P, C = pillar_points.shape
        device = pillar_points.device

        # Compute pillar centroid
        mask = torch.arange(P, device=device).unsqueeze(0) < num_points_per_pillar.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()  # (N, P, 1)

        points_sum = (pillar_points * mask_f).sum(dim=1, keepdim=True)
        counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        centroid = points_sum / counts  # (N, 1, C)

        # Offset from centroid
        centroid_offset = pillar_points[:, :, :3] - centroid[:, :, :3]

        # Pillar center in world coords (use first 2 coords)
        pillar_center = pillar_coords.float().unsqueeze(1).expand(-1, P, -1)

        # Augmented features
        augmented = torch.cat([
            pillar_points,
            centroid_offset,
            pillar_center,
        ], dim=-1)  # (N, P, C+5)

        # Shared MLP
        x = self.linear(augmented)  # (N, P, D)
        x = x.permute(0, 2, 1)  # (N, D, P)

        # Handle batch norm with reshape
        N_pil, D, P_dim = x.shape
        x = x.reshape(N_pil * D, P_dim).unsqueeze(-1)
        x = x.reshape(N_pil, D, P_dim)

        x = F.relu(x)

        # Mask and max-pool
        x = x * mask.unsqueeze(1).float()
        x = x.max(dim=2)[0]  # (N, D)

        return x


class PillarScatter(nn.Module):
    """Scatters pillar features onto a 2D pseudo-image (BEV grid)."""

    def __init__(self, feature_channels: int = 64, grid_size: tuple[int, int] = (500, 500)):
        super().__init__()
        self.feature_channels = feature_channels
        self.grid_h, self.grid_w = grid_size

    def forward(
        self, pillar_features: torch.Tensor, pillar_coords: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Scatter pillar features to pseudo-image.

        Args:
            pillar_features: (N_total_pillars, D)
            pillar_coords: (N_total_pillars, 3) — (batch_idx, y_idx, x_idx)
            batch_size: number of samples in batch

        Returns:
            Pseudo-image (B, D, H, W)
        """
        D = self.feature_channels
        device = pillar_features.device
        canvas = torch.zeros(batch_size, D, self.grid_h, self.grid_w, device=device)

        batch_idx = pillar_coords[:, 0].long()
        y_idx = pillar_coords[:, 1].long().clamp(0, self.grid_h - 1)
        x_idx = pillar_coords[:, 2].long().clamp(0, self.grid_w - 1)

        canvas[batch_idx, :, y_idx, x_idx] = pillar_features

        return canvas


class PointPillarsBackbone(nn.Module):
    """2D CNN backbone operating on pseudo-image."""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.block1 = self._make_block(in_channels, 64, 3, stride=2, num_layers=4)
        self.block2 = self._make_block(64, 128, 3, stride=2, num_layers=6)
        self.block3 = self._make_block(128, 256, 3, stride=2, num_layers=6)

        # Upsample and concatenate
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def _make_block(
        self, in_ch: int, out_ch: int, kernel: int, stride: int, num_layers: int
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        up1 = self.up1(x1)
        up2 = self.up2(x2)
        up3 = self.up3(x3)

        # Match spatial dimensions
        target_h, target_w = up1.shape[2], up1.shape[3]
        up2 = F.interpolate(up2, size=(target_h, target_w), mode="bilinear", align_corners=False)
        up3 = F.interpolate(up3, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return torch.cat([up1, up2, up3], dim=1)  # (B, 384, H/2, W/2)


class PointPillarsDetector(nn.Module):
    """Complete PointPillars 3D object detector."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        in_channels = config.get("input_channels", 4)
        pillar_features = config.get("pillar_features", 64)
        max_points = config.get("max_points_per_voxel", 32)
        grid_size = tuple(config.get("grid_size", [500, 500]))
        num_classes = config.get("num_classes", 3)

        self.pillar_net = PillarFeatureNet(in_channels, pillar_features, max_points)
        self.scatter = PillarScatter(pillar_features, grid_size)
        self.backbone = PointPillarsBackbone(pillar_features)
        self.head = DetectionHead(in_channels=384, num_classes=num_classes)

        self.score_threshold = config.get("score_threshold", 0.3)
        self.nms_threshold = config.get("nms_threshold", 0.5)

    def forward(
        self,
        pillar_points: torch.Tensor,
        pillar_coords: torch.Tensor,
        num_points_per_pillar: torch.Tensor,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Run PointPillars detection.

        Args:
            pillar_points: (N_pillars, max_points, C)
            pillar_coords: (N_pillars, 3) — (batch_idx, y, x)
            num_points_per_pillar: (N_pillars,)
            batch_size: batch size

        Returns:
            Dictionary with 'cls_scores', 'bbox_preds', 'dir_preds'
        """
        pillar_feat = self.pillar_net(pillar_points, pillar_coords[:, 1:], num_points_per_pillar)
        pseudo_image = self.scatter(pillar_feat, pillar_coords, batch_size)
        features = self.backbone(pseudo_image)
        predictions = self.head(features)
        return predictions

    @torch.no_grad()
    def predict(
        self,
        pillar_points: torch.Tensor,
        pillar_coords: torch.Tensor,
        num_points_per_pillar: torch.Tensor,
        batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Run inference with post-processing (NMS).

        Returns:
            List of dicts per batch sample: {'boxes', 'scores', 'labels'}
        """
        self.eval()
        preds = self.forward(pillar_points, pillar_coords, num_points_per_pillar, batch_size)

        cls_scores = torch.sigmoid(preds["cls_scores"])
        bbox_preds = preds["bbox_preds"]
        B = cls_scores.shape[0]

        results = []
        for b in range(B):
            scores = cls_scores[b]  # (num_classes, H, W)
            boxes = bbox_preds[b]  # (7, H, W)

            num_cls = scores.shape[0]
            H, W = scores.shape[1], scores.shape[2]

            scores_flat = scores.reshape(num_cls, -1).t()  # (H*W, num_cls)
            boxes_flat = boxes.reshape(7, -1).t()  # (H*W, 7)

            max_scores, labels = scores_flat.max(dim=1)
            mask = max_scores > self.score_threshold

            if mask.sum() == 0:
                results.append({
                    "boxes": torch.zeros(0, 7, device=scores.device),
                    "scores": torch.zeros(0, device=scores.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=scores.device),
                })
                continue

            filtered_scores = max_scores[mask]
            filtered_boxes = boxes_flat[mask]
            filtered_labels = labels[mask]

            keep = nms_3d(filtered_boxes, filtered_scores, self.nms_threshold)
            results.append({
                "boxes": filtered_boxes[keep],
                "scores": filtered_scores[keep],
                "labels": filtered_labels[keep],
            })

        return results
