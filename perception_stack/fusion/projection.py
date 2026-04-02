"""Camera-LiDAR projection utilities."""

import numpy as np


class ProjectionMatrix:
    """Handles projection between camera, LiDAR, and world coordinate frames."""

    def __init__(
        self,
        camera_intrinsic: np.ndarray,
        camera_extrinsic: np.ndarray,
        lidar_to_camera: np.ndarray | None = None,
    ):
        """Initialize projection matrices.

        Args:
            camera_intrinsic: (3, 3) camera intrinsic matrix
            camera_extrinsic: (4, 4) camera-to-world transform
            lidar_to_camera: (4, 4) LiDAR-to-camera transform
        """
        self.K = camera_intrinsic.copy()
        self.cam_to_world = camera_extrinsic.copy()
        self.world_to_cam = np.linalg.inv(camera_extrinsic)

        if lidar_to_camera is not None:
            self.lidar_to_cam = lidar_to_camera.copy()
        else:
            self.lidar_to_cam = np.eye(4)

    def project_lidar_to_image(
        self, points: np.ndarray, image_shape: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project LiDAR points to camera image coordinates.

        Args:
            points: (N, 3) or (N, 4) LiDAR points
            image_shape: (H, W) for filtering out-of-bounds projections

        Returns:
            uv: (M, 2) image coordinates
            depth: (M,) depth values
        """
        N = points.shape[0]
        pts = np.ones((N, 4))
        pts[:, :3] = points[:, :3]

        # LiDAR -> camera
        cam_pts = (self.lidar_to_cam @ pts.T).T  # (N, 4)

        # Filter points behind camera
        depth = cam_pts[:, 2]
        valid = depth > 0.1
        cam_pts = cam_pts[valid]
        depth = depth[valid]

        # Camera -> image
        img_pts = (self.K @ cam_pts[:, :3].T).T  # (N, 3)
        uv = img_pts[:, :2] / (img_pts[:, 2:3] + 1e-8)

        # Filter out-of-bounds
        if image_shape is not None:
            H, W = image_shape
            in_bounds = (
                (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                (uv[:, 1] >= 0) & (uv[:, 1] < H)
            )
            uv = uv[in_bounds]
            depth = depth[in_bounds]

        return uv, depth

    def project_world_to_image(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to image coordinates.

        Args:
            points: (N, 3) world coordinates

        Returns:
            uv: (M, 2) image coordinates
            depth: (M,) depth values
        """
        N = points.shape[0]
        pts = np.ones((N, 4))
        pts[:, :3] = points

        cam_pts = (self.world_to_cam @ pts.T).T
        depth = cam_pts[:, 2]
        valid = depth > 0.1

        img_pts = (self.K @ cam_pts[valid, :3].T).T
        uv = img_pts[:, :2] / (img_pts[:, 2:3] + 1e-8)

        return uv, depth[valid]

    def unproject_image_to_world(
        self, uv: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """Unproject image pixels with depth to world coordinates.

        Args:
            uv: (N, 2) pixel coordinates
            depth: (N,) depth values

        Returns:
            (N, 3) world coordinates
        """
        K_inv = np.linalg.inv(self.K)
        N = uv.shape[0]
        pixels = np.ones((N, 3))
        pixels[:, :2] = uv

        cam_rays = (K_inv @ pixels.T).T  # (N, 3)
        cam_pts = cam_rays * depth[:, None]

        pts_h = np.ones((N, 4))
        pts_h[:, :3] = cam_pts
        world_pts = (self.cam_to_world @ pts_h.T).T[:, :3]

        return world_pts
