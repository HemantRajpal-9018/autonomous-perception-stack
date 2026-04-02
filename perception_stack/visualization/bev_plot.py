"""Bird's Eye View plot visualization."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CLASS_COLORS = {0: "green", 1: "yellow", 2: "cyan"}
CLASS_NAMES = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}


def plot_bev(
    boxes: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    point_cloud: np.ndarray | None = None,
    occupancy_grid: np.ndarray | None = None,
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
    title: str = "Bird's Eye View",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Create BEV visualization plot.

    Args:
        boxes: (N, 7) bounding boxes
        labels: (N,) class labels
        point_cloud: (M, 3+) point cloud
        occupancy_grid: (H, W) occupancy probabilities
        xlim, ylim: axis limits in meters
        title: plot title
        save_path: optional path to save figure
        ax: optional existing axes

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.get_figure()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Occupancy grid
    if occupancy_grid is not None:
        ax.imshow(
            occupancy_grid,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            cmap="RdYlGn_r",
            alpha=0.5,
            vmin=0,
            vmax=1,
        )

    # Point cloud
    if point_cloud is not None:
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            s=0.5,
            c="gray",
            alpha=0.3,
        )

    # Bounding boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            x, y, _, w, l, _, yaw = box
            label = int(labels[i]) if labels is not None else 0
            color = CLASS_COLORS.get(label, "white")

            # Rotated rectangle
            corners = _box_corners_bev(x, y, w, l, yaw)
            polygon = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(polygon)

            # Direction arrow
            dx = np.cos(yaw) * l / 2
            dy = np.sin(yaw) * l / 2
            ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2, fc=color, ec=color)

    # Ego vehicle
    ego = patches.Rectangle((-1, -2), 2, 4, fill=True, facecolor="blue", alpha=0.5)
    ax.add_patch(ego)
    ax.text(0, 0, "EGO", ha="center", va="center", fontsize=8, color="white")

    # Legend
    legend_handles = [
        patches.Patch(color=c, label=CLASS_NAMES.get(k, "unknown"))
        for k, c in CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _box_corners_bev(
    x: float, y: float, w: float, l: float, yaw: float
) -> np.ndarray:
    """Get 4 BEV corners of a rotated rectangle."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    dx = np.array([w / 2, w / 2, -w / 2, -w / 2])
    dy = np.array([l / 2, -l / 2, -l / 2, l / 2])

    corners_x = cos_yaw * dx - sin_yaw * dy + x
    corners_y = sin_yaw * dx + cos_yaw * dy + y

    return np.stack([corners_x, corners_y], axis=1)
