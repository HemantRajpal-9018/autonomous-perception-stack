"""Camera-LiDAR late fusion for 3D object detection."""

import numpy as np
import torch
import torch.nn as nn

from perception_stack.fusion.projection import ProjectionMatrix


class FeatureFusionMLP(nn.Module):
    """MLP to fuse camera and LiDAR features per detection."""

    def __init__(self, camera_feat_dim: int = 64, lidar_feat_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(camera_feat_dim + lidar_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, camera_features: torch.Tensor, lidar_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([camera_features, lidar_features], dim=-1)
        return self.net(combined)


class CameraLiDARFusion:
    """Late fusion of camera and LiDAR detections.

    Associates camera detections with LiDAR detections based on
    spatial proximity after projecting to a common frame.
    """

    def __init__(
        self,
        camera_weight: float = 0.4,
        lidar_weight: float = 0.6,
        association_threshold: float = 2.0,
    ):
        self.camera_weight = camera_weight
        self.lidar_weight = lidar_weight
        self.association_threshold = association_threshold

    def associate_detections(
        self,
        camera_boxes: np.ndarray,
        lidar_boxes: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate camera and LiDAR detections by center distance.

        Args:
            camera_boxes: (N, 7) camera detections
            lidar_boxes: (M, 7) LiDAR detections

        Returns:
            matches, unmatched_camera, unmatched_lidar
        """
        if len(camera_boxes) == 0 or len(lidar_boxes) == 0:
            return (
                [],
                list(range(len(camera_boxes))),
                list(range(len(lidar_boxes))),
            )

        # Distance matrix using XY centers
        cam_centers = camera_boxes[:, :2]
        lid_centers = lidar_boxes[:, :2]

        dist_matrix = np.linalg.norm(
            cam_centers[:, None, :] - lid_centers[None, :, :], axis=-1
        )

        # Greedy matching
        matches = []
        used_cam = set()
        used_lid = set()

        flat_indices = np.argsort(dist_matrix.ravel())
        N = len(camera_boxes)

        for flat_idx in flat_indices:
            ci = flat_idx // len(lidar_boxes)
            li = flat_idx % len(lidar_boxes)

            if ci in used_cam or li in used_lid:
                continue
            if dist_matrix[ci, li] > self.association_threshold:
                break

            matches.append((ci, li))
            used_cam.add(ci)
            used_lid.add(li)

        unmatched_cam = [i for i in range(N) if i not in used_cam]
        unmatched_lid = [i for i in range(len(lidar_boxes)) if i not in used_lid]

        return matches, unmatched_cam, unmatched_lid

    def fuse_boxes(
        self,
        camera_boxes: np.ndarray,
        camera_scores: np.ndarray,
        lidar_boxes: np.ndarray,
        lidar_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse camera and LiDAR detections.

        Matched pairs are merged via weighted average.
        Unmatched detections are kept.

        Args:
            camera_boxes: (N, 7)
            camera_scores: (N,)
            lidar_boxes: (M, 7)
            lidar_scores: (M,)

        Returns:
            fused_boxes: (K, 7)
            fused_scores: (K,)
        """
        matches, unmatched_cam, unmatched_lid = self.associate_detections(
            camera_boxes, lidar_boxes
        )

        fused_boxes = []
        fused_scores = []

        # Fused detections
        for ci, li in matches:
            wc = self.camera_weight * camera_scores[ci]
            wl = self.lidar_weight * lidar_scores[li]
            total_w = wc + wl + 1e-8

            fused_box = (wc * camera_boxes[ci] + wl * lidar_boxes[li]) / total_w
            fused_score = max(camera_scores[ci], lidar_scores[li])

            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)

        # Add unmatched camera detections
        for ci in unmatched_cam:
            fused_boxes.append(camera_boxes[ci])
            fused_scores.append(camera_scores[ci] * self.camera_weight)

        # Add unmatched LiDAR detections
        for li in unmatched_lid:
            fused_boxes.append(lidar_boxes[li])
            fused_scores.append(lidar_scores[li] * self.lidar_weight)

        if len(fused_boxes) == 0:
            return np.zeros((0, 7)), np.zeros(0)

        return np.array(fused_boxes), np.array(fused_scores)

    def paint_lidar_with_camera(
        self,
        point_cloud: np.ndarray,
        image: np.ndarray,
        projection: ProjectionMatrix,
    ) -> np.ndarray:
        """Augment LiDAR points with camera RGB features.

        Projects each LiDAR point to the image and samples the pixel color.

        Args:
            point_cloud: (N, 4) — x, y, z, intensity
            image: (H, W, 3) camera image
            projection: ProjectionMatrix instance

        Returns:
            (N, 7) — x, y, z, intensity, r, g, b
        """
        H, W = image.shape[:2]
        uv, depth = projection.project_lidar_to_image(point_cloud[:, :3], (H, W))

        # Initialize painted cloud with zeros for color
        painted = np.zeros((len(point_cloud), 7))
        painted[:, :4] = point_cloud[:, :4] if point_cloud.shape[1] >= 4 else np.hstack([
            point_cloud[:, :3], np.zeros((len(point_cloud), 1))
        ])

        # Map valid projections back — we need the original indices
        # Recompute to get validity masks
        N = point_cloud.shape[0]
        pts = np.ones((N, 4))
        pts[:, :3] = point_cloud[:, :3]
        cam_pts = (projection.lidar_to_cam @ pts.T).T
        z = cam_pts[:, 2]
        valid_depth = z > 0.1

        img_pts = (projection.K @ cam_pts[valid_depth, :3].T).T
        uv_all = img_pts[:, :2] / (img_pts[:, 2:3] + 1e-8)

        in_bounds = (
            (uv_all[:, 0] >= 0) & (uv_all[:, 0] < W) &
            (uv_all[:, 1] >= 0) & (uv_all[:, 1] < H)
        )

        valid_indices = np.where(valid_depth)[0][in_bounds]
        valid_uv = uv_all[in_bounds].astype(int)

        painted[valid_indices, 4:7] = image[valid_uv[:, 1], valid_uv[:, 0]] / 255.0

        return painted
