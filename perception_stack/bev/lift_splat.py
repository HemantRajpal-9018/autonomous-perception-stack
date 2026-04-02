"""Lift-Splat-Shoot BEV feature generation.

Reference: Philion & Fidler, 'Lift, Splat, Shoot' (ECCV 2020).
Lifts 2D image features to 3D using predicted depth, then splats onto BEV grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNet(nn.Module):
    """Predicts discrete depth distribution for each pixel."""

    def __init__(self, in_channels: int, num_depth_bins: int = 41):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_depth_bins, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LiftSplatShoot(nn.Module):
    """Lift-Splat-Shoot BEV encoder.

    1. Extract image features with a CNN backbone
    2. Predict depth distribution per pixel (Lift)
    3. Create outer product of features and depth (frustum features)
    4. Project frustum points to BEV grid (Splat)
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 400),
        bev_size: tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,
        feature_channels: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 50.0,
        num_depth_bins: int = 41,
        downsample_factor: int = 16,
    ):
        super().__init__()
        self.image_size = image_size
        self.bev_size = bev_size
        self.bev_resolution = bev_resolution
        self.feature_channels = feature_channels
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depth_bins = num_depth_bins
        self.ds = downsample_factor

        # Image feature backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.depth_net = DepthNet(feature_channels, num_depth_bins)

        # BEV encoder
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # Pre-compute depth bins
        self.register_buffer(
            "depth_bins",
            torch.linspace(depth_min, depth_max, num_depth_bins),
        )

    def create_frustum(self, feat_h: int, feat_w: int, device: torch.device) -> torch.Tensor:
        """Create frustum point cloud for lifting.

        Returns:
            Frustum grid (D, H, W, 3) with (u, v, depth) coordinates
        """
        D = self.num_depth_bins
        ds = self.ds

        # Image pixel coordinates at feature resolution
        us = torch.linspace(0, self.image_size[1] - 1, feat_w, device=device)
        vs = torch.linspace(0, self.image_size[0] - 1, feat_h, device=device)
        depths = self.depth_bins

        grid_d, grid_v, grid_u = torch.meshgrid(depths, vs, us, indexing="ij")
        frustum = torch.stack([grid_u, grid_v, grid_d], dim=-1)  # (D, H, W, 3)
        return frustum

    def lift(
        self,
        features: torch.Tensor,
        depth_probs: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift 2D features to 3D using depth.

        Args:
            features: (B, C, H, W)
            depth_probs: (B, D, H, W)
            intrinsics: (B, 3, 3)
            extrinsics: (B, 4, 4)

        Returns:
            points_3d: (B, N, 3) world coordinates
            point_features: (B, N, C) per-point features
        """
        B, C, H, W = features.shape
        D = depth_probs.shape[1]
        device = features.device

        frustum = self.create_frustum(H, W, device)  # (D, H, W, 3)

        # Unproject: pixel (u,v,d) -> camera (x,y,z)
        u = frustum[..., 0]
        v = frustum[..., 1]
        d = frustum[..., 2]

        fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)

        x_cam = (u.unsqueeze(0) - cx) * d.unsqueeze(0) / (fx + 1e-8)
        y_cam = (v.unsqueeze(0) - cy) * d.unsqueeze(0) / (fy + 1e-8)
        z_cam = d.unsqueeze(0).expand(B, -1, -1, -1)

        cam_pts = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)
        cam_pts_flat = cam_pts.reshape(B, -1, 4)  # (B, D*H*W, 4)

        # Camera to world transform
        extrinsics_inv = torch.inverse(extrinsics)
        world_pts = (extrinsics_inv[:, :3, :4] @ cam_pts_flat.transpose(1, 2)).transpose(1, 2)

        # Outer product: depth_probs x features
        depth_probs_softmax = F.softmax(depth_probs, dim=1)
        feat_expanded = features.unsqueeze(2).expand(-1, -1, D, -1, -1)  # (B, C, D, H, W)
        depth_expanded = depth_probs_softmax.unsqueeze(1)  # (B, 1, D, H, W)
        lifted = (feat_expanded * depth_expanded).reshape(B, C, -1)  # (B, C, D*H*W)

        return world_pts, lifted.transpose(1, 2)

    def splat(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Splat 3D point features onto BEV grid.

        Args:
            points: (B, N, 3) world coordinates
            features: (B, N, C) point features

        Returns:
            BEV grid (B, C, bev_h, bev_w)
        """
        B, N, C = features.shape
        bev_h, bev_w = self.bev_size
        device = features.device

        bev_grid = torch.zeros(B, C, bev_h, bev_w, device=device)

        # Convert world XY to BEV grid indices
        x_idx = ((points[:, :, 0] / self.bev_resolution) + bev_w / 2).long()
        y_idx = ((points[:, :, 1] / self.bev_resolution) + bev_h / 2).long()

        # Mask valid indices
        valid = (x_idx >= 0) & (x_idx < bev_w) & (y_idx >= 0) & (y_idx < bev_h)

        for b in range(B):
            mask = valid[b]
            xi = x_idx[b][mask]
            yi = y_idx[b][mask]
            feats = features[b][mask]  # (M, C)
            flat_idx = yi * bev_w + xi  # (M,)
            bev_flat = bev_grid[b].reshape(C, -1)  # (C, H*W)
            bev_flat.scatter_add_(1, flat_idx.unsqueeze(0).expand(C, -1), feats.t())

        return bev_grid

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end: multi-cam images to BEV features.

        Args:
            images: (B, N_cam, 3, H, W)
            intrinsics: (B, N_cam, 3, 3)
            extrinsics: (B, N_cam, 4, 4)

        Returns:
            BEV feature map (B, C, bev_h, bev_w)
        """
        B, N_cam = images.shape[:2]
        bev_h, bev_w = self.bev_size

        bev_accum = torch.zeros(
            B, self.feature_channels, bev_h, bev_w, device=images.device
        )

        for cam_idx in range(N_cam):
            cam_img = images[:, cam_idx]
            features = self.backbone(cam_img)
            depth_logits = self.depth_net(features)

            pts, feats = self.lift(
                features, depth_logits, intrinsics[:, cam_idx], extrinsics[:, cam_idx]
            )
            bev = self.splat(pts, feats)
            bev_accum = bev_accum + bev

        bev_out = self.bev_encoder(bev_accum)
        return bev_out
