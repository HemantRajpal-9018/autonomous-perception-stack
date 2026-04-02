"""Shared test fixtures for perception stack tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_point_cloud():
    """Generate a random point cloud (N, 4) — x, y, z, intensity."""
    np.random.seed(42)
    N = 1000
    points = np.random.randn(N, 4).astype(np.float32)
    points[:, :3] *= 20  # Scale spatial coordinates
    points[:, 3] = np.abs(points[:, 3])  # Positive intensity
    return points


@pytest.fixture
def sample_boxes():
    """Generate sample 3D bounding boxes (N, 7)."""
    return np.array([
        [10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.1],   # vehicle
        [-5.0, 3.0, 0.0, 0.6, 0.6, 1.8, 0.0],    # pedestrian
        [15.0, -2.0, 0.0, 0.8, 1.8, 1.5, -0.5],  # cyclist
        [20.0, 10.0, 0.0, 2.0, 5.0, 2.0, 0.3],   # vehicle
    ], dtype=np.float32)


@pytest.fixture
def sample_labels():
    return np.array([0, 1, 2, 0], dtype=np.int64)


@pytest.fixture
def sample_scores():
    return np.array([0.95, 0.87, 0.76, 0.60], dtype=np.float32)


@pytest.fixture
def sample_image():
    """Generate a random image (3, H, W)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 400)


@pytest.fixture
def sample_intrinsic():
    """Camera intrinsic matrix."""
    return np.array([
        [721.5, 0.0, 609.5],
        [0.0, 721.5, 172.8],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def sample_extrinsic():
    """Camera extrinsic (identity = camera at world origin)."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def detection_config():
    """Standard detection config."""
    return {
        "num_classes": 3,
        "class_names": ["vehicle", "pedestrian", "cyclist"],
        "score_threshold": 0.3,
        "nms_threshold": 0.5,
        "input_channels": 4,
        "pillar_features": 64,
        "grid_size": [100, 100],
        "max_points_per_voxel": 32,
    }


@pytest.fixture
def tracking_config():
    """Standard tracking config."""
    return {
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
        "dt": 0.1,
        "process_noise": 1.0,
        "measurement_noise": 0.5,
    }
