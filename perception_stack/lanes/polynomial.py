"""Polynomial curve fitting for lane lines."""

import numpy as np
import torch


class PolynomialLaneFitter:
    """Fits polynomial curves to lane line pixel coordinates.

    Extracts lane pixels from a segmentation mask, clusters them by
    horizontal position, and fits a polynomial to each cluster.
    """

    def __init__(self, degree: int = 3, num_lanes: int = 4, image_height: int = 720,
                 min_points: int = 50):
        self.degree = degree
        self.num_lanes = num_lanes
        self.image_height = image_height
        self.min_points = min_points

    def extract_lane_pixels(
        self, seg_mask: np.ndarray, lane_class_id: int = 1
    ) -> list[np.ndarray]:
        """Extract and cluster lane pixels from segmentation mask.

        Args:
            seg_mask: (H, W) integer mask
            lane_class_id: class index for lane lines

        Returns:
            List of (N, 2) arrays with (y, x) coordinates per lane
        """
        ys, xs = np.where(seg_mask == lane_class_id)
        if len(xs) < self.min_points:
            return []

        # Cluster by x-position using histogram binning
        num_bins = self.num_lanes * 2
        hist, bin_edges = np.histogram(xs, bins=num_bins)

        lanes = []
        for i in range(num_bins):
            mask = (xs >= bin_edges[i]) & (xs < bin_edges[i + 1])
            if mask.sum() >= self.min_points:
                lane_ys = ys[mask]
                lane_xs = xs[mask]
                lanes.append(np.stack([lane_ys, lane_xs], axis=1))

        # Merge close clusters and keep top-N
        merged = self._merge_close_lanes(lanes)
        return merged[:self.num_lanes]

    def _merge_close_lanes(
        self, lanes: list[np.ndarray], threshold: float = 30.0
    ) -> list[np.ndarray]:
        """Merge lane clusters whose mean x-positions are too close."""
        if len(lanes) <= 1:
            return lanes

        # Sort by mean x position
        lanes = sorted(lanes, key=lambda l: l[:, 1].mean())
        merged = [lanes[0]]

        for lane in lanes[1:]:
            prev_mean_x = merged[-1][:, 1].mean()
            curr_mean_x = lane[:, 1].mean()
            if abs(curr_mean_x - prev_mean_x) < threshold:
                merged[-1] = np.concatenate([merged[-1], lane], axis=0)
            else:
                merged.append(lane)

        return merged

    def fit_polynomial(self, lane_points: np.ndarray) -> np.ndarray | None:
        """Fit polynomial to a single lane's points.

        Args:
            lane_points: (N, 2) array with (y, x) coordinates

        Returns:
            Polynomial coefficients [a_n, ..., a_1, a_0] or None if fit fails
        """
        if len(lane_points) < self.degree + 1:
            return None

        y = lane_points[:, 0].astype(np.float64)
        x = lane_points[:, 1].astype(np.float64)

        try:
            coeffs = np.polyfit(y, x, self.degree)
            return coeffs
        except (np.linalg.LinAlgError, ValueError):
            return None

    def evaluate_polynomial(
        self, coeffs: np.ndarray, y_range: np.ndarray | None = None
    ) -> np.ndarray:
        """Evaluate polynomial at given y-coordinates.

        Args:
            coeffs: polynomial coefficients
            y_range: y values to evaluate at (default: full image height)

        Returns:
            (N, 2) array of (y, x) points along the fitted lane
        """
        if y_range is None:
            y_range = np.linspace(0, self.image_height - 1, self.image_height)

        x_values = np.polyval(coeffs, y_range)
        return np.stack([y_range, x_values], axis=1)

    def fit_lanes(self, seg_mask: np.ndarray) -> list[dict]:
        """Complete lane fitting pipeline.

        Args:
            seg_mask: (H, W) segmentation mask

        Returns:
            List of lane dicts with 'coefficients', 'points', 'confidence'
        """
        lane_clusters = self.extract_lane_pixels(seg_mask)
        results = []

        for cluster in lane_clusters:
            coeffs = self.fit_polynomial(cluster)
            if coeffs is None:
                continue

            points = self.evaluate_polynomial(coeffs)
            # Confidence based on fit residual
            y = cluster[:, 0].astype(np.float64)
            x = cluster[:, 1].astype(np.float64)
            predicted_x = np.polyval(coeffs, y)
            residual = np.mean(np.abs(x - predicted_x))
            confidence = max(0.0, 1.0 - residual / 50.0)

            results.append({
                "coefficients": coeffs,
                "points": points,
                "confidence": confidence,
                "num_pixels": len(cluster),
            })

        return results

    def fit_lanes_torch(self, seg_mask: torch.Tensor) -> list[dict]:
        """Torch-compatible wrapper for fit_lanes."""
        if seg_mask.dim() == 3:
            seg_mask = seg_mask.argmax(dim=0)
        return self.fit_lanes(seg_mask.cpu().numpy())
