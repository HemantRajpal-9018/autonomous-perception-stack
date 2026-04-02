"""Tracking visualization — shows track trajectories and IDs."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CLASS_COLORS = {0: "green", 1: "yellow", 2: "cyan"}


def plot_tracking(
    tracks: list,
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
    title: str = "Multi-Object Tracking",
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize tracked objects with trajectories.

    Args:
        tracks: List of Track objects with bbox_3d, track_id, label, history
        xlim, ylim: axis limits
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
    ax.grid(True, alpha=0.3)

    # Ego vehicle
    ego = patches.Rectangle((-1, -2), 2, 4, fill=True, facecolor="blue", alpha=0.5)
    ax.add_patch(ego)

    for track in tracks:
        bbox = track.bbox_3d
        x, y = bbox[0], bbox[1]
        w, l, yaw = bbox[3], bbox[4], bbox[6]
        color = CLASS_COLORS.get(track.label, "white")

        # Current bounding box
        corners = _box_corners(x, y, w, l, yaw)
        polygon = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(polygon)

        # Track ID label
        ax.text(x, y + l / 2 + 0.5, f"ID:{track.track_id}",
                ha="center", fontsize=8, color=color,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

        # Trajectory history
        if hasattr(track, "history") and len(track.history) > 1:
            history = np.array(track.history)
            ax.plot(history[:, 0], history[:, 1], "-", color=color, alpha=0.5, linewidth=1)
            ax.scatter(history[-1, 0], history[-1, 1], c=color, s=20, zorder=5)

        # Velocity arrow
        if hasattr(track, "velocity"):
            vel = track.velocity
            speed = np.linalg.norm(vel[:2])
            if speed > 0.5:
                ax.arrow(x, y, vel[0], vel[1],
                        head_width=0.3, head_length=0.2, fc=color, ec=color, alpha=0.7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _box_corners(x: float, y: float, w: float, l: float, yaw: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    dx = np.array([w / 2, w / 2, -w / 2, -w / 2])
    dy = np.array([l / 2, -l / 2, -l / 2, l / 2])
    corners_x = cos_yaw * dx - sin_yaw * dy + x
    corners_y = sin_yaw * dx + cos_yaw * dy + y
    return np.stack([corners_x, corners_y], axis=1)
