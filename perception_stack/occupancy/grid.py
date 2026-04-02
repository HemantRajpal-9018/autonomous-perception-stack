"""Occupancy grid for environment representation.

Maintains a probabilistic 2D grid where each cell stores the log-odds
probability of being occupied. Updated via inverse sensor models from
LiDAR scans and tracked object positions.
"""

import numpy as np


class OccupancyGrid:
    """2D probabilistic occupancy grid for motion planning handoff.

    Each cell stores log-odds probability:
      l = log(p / (1-p))
    Updated using Bayesian updates with sensor observations.
    """

    def __init__(
        self,
        width: int = 200,
        height: int = 200,
        resolution: float = 0.5,
        origin: tuple[float, float] = (-50.0, -50.0),
        occupied_threshold: float = 0.7,
        free_threshold: float = 0.3,
        decay_rate: float = 0.95,
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin)
        self.occupied_threshold = occupied_threshold
        self.free_threshold = free_threshold
        self.decay_rate = decay_rate

        # Log-odds grid (initialized to 0 = unknown = p=0.5)
        self.log_odds = np.zeros((height, width), dtype=np.float64)

        # Sensor model parameters
        self.l_occ = np.log(0.7 / 0.3)  # log-odds for occupied
        self.l_free = np.log(0.3 / 0.7)  # log-odds for free

        # Clamp bounds to avoid numerical issues
        self.l_max = 5.0
        self.l_min = -5.0

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return x, y

    def is_valid(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def _bresenham_line(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[tuple[int, int]]:
        """Bresenham's line algorithm for ray tracing."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells

    def update_from_lidar(
        self, point_cloud: np.ndarray, sensor_position: np.ndarray
    ) -> None:
        """Update grid from LiDAR point cloud using ray tracing.

        Args:
            point_cloud: (N, 3+) point cloud, uses XY only
            sensor_position: (2,) or (3,) sensor position in world frame
        """
        sx, sy = self.world_to_grid(sensor_position[0], sensor_position[1])

        for i in range(len(point_cloud)):
            px, py = self.world_to_grid(point_cloud[i, 0], point_cloud[i, 1])

            if not self.is_valid(px, py):
                continue

            # Ray trace from sensor to point — free cells
            ray_cells = self._bresenham_line(sx, sy, px, py)
            for gx, gy in ray_cells[:-1]:  # All but endpoint are free
                if self.is_valid(gx, gy):
                    self.log_odds[gy, gx] += self.l_free
                    self.log_odds[gy, gx] = np.clip(
                        self.log_odds[gy, gx], self.l_min, self.l_max
                    )

            # Endpoint is occupied
            self.log_odds[py, px] += self.l_occ
            self.log_odds[py, px] = np.clip(
                self.log_odds[py, px], self.l_min, self.l_max
            )

    def update_from_detections(self, boxes: np.ndarray) -> None:
        """Mark cells occupied based on detected bounding boxes.

        Args:
            boxes: (N, 7) — x, y, z, w, l, h, yaw
        """
        for box in boxes:
            cx, cy = box[0], box[1]
            w, l = box[3], box[4]

            # Axis-aligned approximation
            x_min = cx - w / 2
            x_max = cx + w / 2
            y_min = cy - l / 2
            y_max = cy + l / 2

            gx_min, gy_min = self.world_to_grid(x_min, y_min)
            gx_max, gy_max = self.world_to_grid(x_max, y_max)

            gx_min = max(0, gx_min)
            gy_min = max(0, gy_min)
            gx_max = min(self.width - 1, gx_max)
            gy_max = min(self.height - 1, gy_max)

            self.log_odds[gy_min:gy_max + 1, gx_min:gx_max + 1] += self.l_occ
            self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

    def decay(self) -> None:
        """Apply temporal decay toward unknown (0)."""
        self.log_odds *= self.decay_rate

    @property
    def probability_grid(self) -> np.ndarray:
        """Convert log-odds to probabilities."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    @property
    def occupied_mask(self) -> np.ndarray:
        """Binary mask of occupied cells."""
        return self.probability_grid > self.occupied_threshold

    @property
    def free_mask(self) -> np.ndarray:
        """Binary mask of free cells."""
        return self.probability_grid < self.free_threshold

    def get_costmap(self) -> np.ndarray:
        """Generate costmap for motion planning (0=free, 1=occupied).

        Returns:
            (H, W) float array in [0, 1]
        """
        return self.probability_grid

    def reset(self) -> None:
        self.log_odds.fill(0.0)
