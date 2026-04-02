"""3D bounding box visualization."""

import numpy as np
import cv2


def corners_from_bbox3d(box: np.ndarray) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box.

    Args:
        box: [x, y, z, w, l, h, yaw]

    Returns:
        (8, 3) corner coordinates
    """
    x, y, z, w, l, h, yaw = box

    # Corner offsets (before rotation)
    dx = np.array([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2])
    dy = np.array([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])

    # Rotation around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    corners_x = cos_yaw * dx - sin_yaw * dy + x
    corners_y = sin_yaw * dx + cos_yaw * dy + y
    corners_z = dz + z

    return np.stack([corners_x, corners_y, corners_z], axis=1)


def project_corners_to_image(
    corners: np.ndarray, projection_matrix: np.ndarray
) -> np.ndarray | None:
    """Project 3D corners to 2D image coordinates.

    Args:
        corners: (8, 3) 3D corner coordinates
        projection_matrix: (3, 4) projection matrix

    Returns:
        (8, 2) image coordinates or None if behind camera
    """
    pts = np.hstack([corners, np.ones((8, 1))])
    img_pts = (projection_matrix @ pts.T).T

    # Check if any point is behind the camera
    if np.any(img_pts[:, 2] <= 0):
        return None

    uv = img_pts[:, :2] / img_pts[:, 2:3]
    return uv


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
]

CLASS_COLORS = {
    0: (0, 255, 0),    # vehicle — green
    1: (255, 255, 0),  # pedestrian — yellow
    2: (0, 255, 255),  # cyclist — cyan
}


def draw_3d_boxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray | None = None,
    projection_matrix: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Draw 3D bounding boxes on an image.

    Args:
        image: (H, W, 3) BGR image
        boxes: (N, 7) bounding boxes
        labels: (N,) class labels
        projection_matrix: (3, 4) for projecting 3D to 2D
        scores: (N,) confidence scores

    Returns:
        Image with drawn boxes
    """
    img = image.copy()

    if projection_matrix is None:
        projection_matrix = np.eye(3, 4)

    for i, box in enumerate(boxes):
        corners = corners_from_bbox3d(box)
        uv = project_corners_to_image(corners, projection_matrix)

        if uv is None:
            continue

        label = int(labels[i]) if labels is not None else 0
        color = CLASS_COLORS.get(label, (255, 255, 255))

        pts = uv.astype(int)
        for e0, e1 in BOX_EDGES:
            cv2.line(img, tuple(pts[e0]), tuple(pts[e1]), color, 2)

        # Draw front face with thicker lines
        for e0, e1 in [(0, 1), (0, 4), (1, 5), (4, 5)]:
            cv2.line(img, tuple(pts[e0]), tuple(pts[e1]), color, 3)

        # Score text
        if scores is not None:
            txt = f"{scores[i]:.2f}"
            cv2.putText(img, txt, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def draw_3d_boxes(
    boxes: np.ndarray,
    labels: np.ndarray | None = None,
    image_size: tuple[int, int] = (720, 1280),
) -> np.ndarray:
    """Draw 3D boxes on a blank canvas (for visualization without camera image).

    Args:
        boxes: (N, 7)
        labels: (N,)
        image_size: (H, W)

    Returns:
        (H, W, 3) image
    """
    img = np.zeros((*image_size, 3), dtype=np.uint8)
    # Simple top-down projection
    P = np.array([
        [20.0, 0.0, 0.0, image_size[1] / 2],
        [0.0, 0.0, -20.0, image_size[0] / 2],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return draw_3d_boxes_on_image(img, boxes, labels, P)
