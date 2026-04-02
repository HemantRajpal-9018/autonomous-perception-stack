"""Point cloud visualization utilities."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_point_cloud_bev(
    points: np.ndarray,
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
    point_size: float = 0.3,
    color_by: str = "height",
    title: str = "Point Cloud BEV",
    save_path: str | None = None,
) -> plt.Figure:
    """Render point cloud from bird's eye view.

    Args:
        points: (N, 3+) point cloud
        xlim, ylim: axis limits
        point_size: scatter point size
        color_by: 'height', 'intensity', or 'distance'
        title: plot title
        save_path: optional save path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_facecolor("black")

    # Filter to view range
    mask = (
        (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
        (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1])
    )
    pts = points[mask]

    if len(pts) == 0:
        return fig

    if color_by == "height" and pts.shape[1] >= 3:
        colors = pts[:, 2]
        cmap = "viridis"
    elif color_by == "intensity" and pts.shape[1] >= 4:
        colors = pts[:, 3]
        cmap = "hot"
    elif color_by == "distance":
        colors = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        cmap = "plasma"
    else:
        colors = pts[:, 2] if pts.shape[1] >= 3 else np.zeros(len(pts))
        cmap = "viridis"

    scatter = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=colors, cmap=cmap, s=point_size, alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label=color_by, shrink=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def render_point_cloud_3d(
    points: np.ndarray,
    boxes: np.ndarray | None = None,
    title: str = "3D Point Cloud",
    save_path: str | None = None,
) -> plt.Figure:
    """Render 3D point cloud using matplotlib projection.

    Args:
        points: (N, 3+) point cloud
        boxes: (M, 7) optional bounding boxes
        title: plot title
        save_path: optional save path

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample for performance
    max_pts = 50000
    if len(points) > max_pts:
        idx = np.random.choice(len(points), max_pts, replace=False)
        pts = points[idx]
    else:
        pts = points

    colors = pts[:, 2] if pts.shape[1] >= 3 else np.zeros(len(pts))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, cmap="viridis", s=0.1, alpha=0.5)

    if boxes is not None:
        for box in boxes:
            _draw_box_3d(ax, box)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _draw_box_3d(ax, box: np.ndarray) -> None:
    """Draw a 3D bounding box on a 3D axes."""
    x, y, z, w, l, h, yaw = box

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    dx = np.array([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2])
    dy = np.array([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2])
    dz = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])

    cx = cos_yaw * dx - sin_yaw * dy + x
    cy = sin_yaw * dx + cos_yaw * dy + y
    cz = dz + z

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot([cx[i], cx[j]], [cy[i], cy[j]], [cz[i], cz[j]], "g-", linewidth=1)
