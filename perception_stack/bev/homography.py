"""Homography-based BEV projection from multi-camera inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HomographyBEV(nn.Module):
    """Projects multi-camera images to Bird's Eye View using homography matrices.

    Given camera intrinsics and extrinsics, computes a homography that maps
    image pixels to a ground-plane BEV grid. Features from each camera are
    warped and composited into a single BEV feature map.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (900, 1600),
        bev_size: tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,
        feature_channels: int = 64,
        num_cameras: int = 6,
    ):
        super().__init__()
        self.image_size = image_size
        self.bev_size = bev_size
        self.bev_resolution = bev_resolution
        self.num_cameras = num_cameras

        # Lightweight feature extractor per camera
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # BEV feature fusion
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(feature_channels * num_cameras, feature_channels * 2, 3, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels * 2, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

    def compute_homography(
        self,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        ground_height: float = 0.0,
    ) -> torch.Tensor:
        """Compute homography from image plane to BEV ground plane.

        Args:
            intrinsic: Camera intrinsic matrix (B, 3, 3)
            extrinsic: Camera extrinsic matrix (B, 4, 4) — world-to-camera
            ground_height: Height of the ground plane in world frame

        Returns:
            Homography matrix (B, 3, 3) mapping BEV coords to image coords
        """
        B = intrinsic.shape[0]
        R = extrinsic[:, :3, :3]
        t = extrinsic[:, :3, 3:]

        # Ground plane normal in world frame: [0, 0, 1] at height ground_height
        # Homography H = K * (R - t * n^T / d)
        # For ground plane z=ground_height: n=[0,0,1], d = distance along normal
        n = torch.tensor([0.0, 0.0, 1.0], device=intrinsic.device).view(1, 3, 1).expand(B, -1, -1)
        d = (R @ torch.tensor([0.0, 0.0, ground_height], device=intrinsic.device).view(1, 3, 1)
             + t)[:, 2:3, :]

        d = d.clamp(min=0.1)

        H = intrinsic @ (R - t @ n.transpose(1, 2) / d)
        return H

    def warp_to_bev(
        self, features: torch.Tensor, homography: torch.Tensor
    ) -> torch.Tensor:
        """Warp image features to BEV using homography.

        Args:
            features: Image features (B, C, H_feat, W_feat)
            homography: Homography matrix (B, 3, 3)

        Returns:
            BEV features (B, C, bev_h, bev_w)
        """
        B, C, _, _ = features.shape
        bev_h, bev_w = self.bev_size

        # Create BEV grid coordinates
        ys = torch.linspace(-bev_h // 2, bev_h // 2 - 1, bev_h, device=features.device)
        xs = torch.linspace(-bev_w // 2, bev_w // 2 - 1, bev_w, device=features.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Convert to world coordinates
        world_x = grid_x.flatten() * self.bev_resolution
        world_y = grid_y.flatten() * self.bev_resolution
        ones = torch.ones_like(world_x)
        world_pts = torch.stack([world_x, world_y, ones], dim=0)  # (3, N)

        # Project to image coordinates
        img_pts = homography @ world_pts.unsqueeze(0).expand(B, -1, -1)  # (B, 3, N)
        img_pts = img_pts[:, :2] / (img_pts[:, 2:3] + 1e-8)

        # Normalize to [-1, 1] for grid_sample
        feat_h, feat_w = features.shape[2], features.shape[3]
        img_pts[:, 0] = 2.0 * img_pts[:, 0] / feat_w - 1.0
        img_pts[:, 1] = 2.0 * img_pts[:, 1] / feat_h - 1.0

        grid = img_pts.permute(0, 2, 1).view(B, bev_h, bev_w, 2)

        bev_features = F.grid_sample(
            features, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return bev_features

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: multi-camera images to BEV feature map.

        Args:
            images: (B, N_cam, 3, H, W) multi-camera images
            intrinsics: (B, N_cam, 3, 3) camera intrinsics
            extrinsics: (B, N_cam, 4, 4) camera extrinsics

        Returns:
            BEV feature map (B, C, bev_h, bev_w)
        """
        B, N, C_img, H, W = images.shape
        bev_features_list = []

        for cam_idx in range(N):
            cam_img = images[:, cam_idx]
            cam_feat = self.encoder(cam_img)

            H_mat = self.compute_homography(
                intrinsics[:, cam_idx], extrinsics[:, cam_idx]
            )
            bev_feat = self.warp_to_bev(cam_feat, H_mat)
            bev_features_list.append(bev_feat)

        # Concatenate and fuse
        bev_concat = torch.cat(bev_features_list, dim=1)
        bev_out = self.bev_fusion(bev_concat)
        return bev_out
