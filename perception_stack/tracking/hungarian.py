"""Hungarian algorithm for optimal assignment in multi-object tracking."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_3d_axis_aligned(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute axis-aligned 3D IoU between two sets of boxes.

    Args:
        boxes_a: (N, 7) — x, y, z, w, l, h, yaw
        boxes_b: (M, 7) — x, y, z, w, l, h, yaw

    Returns:
        IoU matrix (N, M)
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    if N == 0 or M == 0:
        return np.zeros((N, M))

    # BEV IoU (xy-plane)
    x1_a = boxes_a[:, 0] - boxes_a[:, 3] / 2
    y1_a = boxes_a[:, 1] - boxes_a[:, 4] / 2
    x2_a = boxes_a[:, 0] + boxes_a[:, 3] / 2
    y2_a = boxes_a[:, 1] + boxes_a[:, 4] / 2

    x1_b = boxes_b[:, 0] - boxes_b[:, 3] / 2
    y1_b = boxes_b[:, 1] - boxes_b[:, 4] / 2
    x2_b = boxes_b[:, 0] + boxes_b[:, 3] / 2
    y2_b = boxes_b[:, 1] + boxes_b[:, 4] / 2

    # Height overlap
    z1_a = boxes_a[:, 2] - boxes_a[:, 5] / 2
    z2_a = boxes_a[:, 2] + boxes_a[:, 5] / 2
    z1_b = boxes_b[:, 2] - boxes_b[:, 5] / 2
    z2_b = boxes_b[:, 2] + boxes_b[:, 5] / 2

    iou_matrix = np.zeros((N, M))

    for i in range(N):
        # BEV intersection
        inter_x1 = np.maximum(x1_a[i], x1_b)
        inter_y1 = np.maximum(y1_a[i], y1_b)
        inter_x2 = np.minimum(x2_a[i], x2_b)
        inter_y2 = np.minimum(y2_a[i], y2_b)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        bev_inter = inter_w * inter_h

        # Height intersection
        inter_z1 = np.maximum(z1_a[i], z1_b)
        inter_z2 = np.minimum(z2_a[i], z2_b)
        height_inter = np.maximum(0, inter_z2 - inter_z1)

        vol_inter = bev_inter * height_inter

        vol_a = boxes_a[i, 3] * boxes_a[i, 4] * boxes_a[i, 5]
        vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]
        vol_union = vol_a + vol_b - vol_inter

        iou_matrix[i] = vol_inter / (vol_union + 1e-8)

    return iou_matrix


def hungarian_assignment(
    cost_matrix: np.ndarray, threshold: float = 0.3
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Perform Hungarian assignment with gating threshold.

    Args:
        cost_matrix: (N, M) cost matrix (lower = better match).
                     Typically 1 - IoU.
        threshold: Maximum cost for valid assignment

    Returns:
        matches: List of (track_idx, detection_idx) pairs
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_detections = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] > threshold:
            continue
        matches.append((r, c))
        unmatched_tracks.discard(r)
        unmatched_detections.discard(c)

    return matches, sorted(unmatched_tracks), sorted(unmatched_detections)
