"""KITTI dataset loader for 3D object detection.

Supports reading:
  - Velodyne point clouds (.bin)
  - Camera images (.png)
  - Calibration files (.txt)
  - Label files (.txt)
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class KITTICalibration:
    """Parser for KITTI calibration files."""

    def __init__(self, calib_path: str | Path):
        self.calib_path = Path(calib_path)
        self._data = {}
        self._parse()

    def _parse(self) -> None:
        with open(self.calib_path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                self._data[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )

    @property
    def P2(self) -> np.ndarray:
        """Camera 2 projection matrix (3, 4)."""
        return self._data.get("P2", np.eye(3, 4)).reshape(3, 4)

    @property
    def R0_rect(self) -> np.ndarray:
        """Rectification rotation (3, 3)."""
        r = self._data.get("R0_rect", np.eye(9)).reshape(3, 3)
        R = np.eye(4)
        R[:3, :3] = r
        return R

    @property
    def Tr_velo_to_cam(self) -> np.ndarray:
        """Velodyne to camera transform (4, 4)."""
        tr = self._data.get("Tr_velo_to_cam", np.eye(12)).reshape(3, 4)
        T = np.eye(4)
        T[:3, :] = tr
        return T

    @property
    def lidar_to_camera(self) -> np.ndarray:
        return self.R0_rect @ self.Tr_velo_to_cam

    @property
    def camera_intrinsic(self) -> np.ndarray:
        """Camera intrinsic matrix (3, 3)."""
        return self.P2[:3, :3]


class KITTILabel:
    """Single KITTI 3D object label."""

    CLASSES = {
        "Car": 0,
        "Van": 0,
        "Truck": 0,
        "Pedestrian": 1,
        "Person_sitting": 1,
        "Cyclist": 2,
        "Tram": 0,
        "Misc": -1,
        "DontCare": -1,
    }

    def __init__(self, line: str):
        parts = line.strip().split()
        self.type = parts[0]
        self.truncation = float(parts[1])
        self.occlusion = int(parts[2])
        self.alpha = float(parts[3])
        self.bbox_2d = np.array([float(x) for x in parts[4:8]])  # x1, y1, x2, y2
        self.h = float(parts[8])
        self.w = float(parts[9])
        self.l = float(parts[10])
        self.location = np.array([float(x) for x in parts[11:14]])  # x, y, z in camera frame
        self.rotation_y = float(parts[14])

    @property
    def class_id(self) -> int:
        return self.CLASSES.get(self.type, -1)

    @property
    def bbox_3d(self) -> np.ndarray:
        """Returns [x, y, z, w, l, h, yaw] in camera frame."""
        return np.array([
            self.location[0], self.location[1], self.location[2],
            self.w, self.l, self.h, self.rotation_y,
        ])


class KITTIDataset(Dataset):
    """KITTI 3D object detection dataset.

    Expected directory structure:
        root/
            velodyne/     — xxxxxx.bin
            image_2/      — xxxxxx.png
            calib/        — xxxxxx.txt
            label_2/      — xxxxxx.txt  (training only)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        max_points: int = 100000,
    ):
        self.root = Path(root_dir)
        self.split = split
        self.max_points = max_points

        # Find available samples
        velodyne_dir = self.root / "velodyne"
        if velodyne_dir.exists():
            self.samples = sorted([p.stem for p in velodyne_dir.glob("*.bin")])
        else:
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def load_point_cloud(self, idx: int) -> np.ndarray:
        """Load Velodyne point cloud.

        Returns:
            (N, 4) array — x, y, z, reflectance
        """
        path = self.root / "velodyne" / f"{self.samples[idx]}.bin"
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        return points

    def load_image(self, idx: int) -> np.ndarray:
        """Load camera image.

        Returns:
            (H, W, 3) uint8 array
        """
        path = self.root / "image_2" / f"{self.samples[idx]}.png"
        return np.array(Image.open(path))

    def load_calibration(self, idx: int) -> KITTICalibration:
        path = self.root / "calib" / f"{self.samples[idx]}.txt"
        return KITTICalibration(path)

    def load_labels(self, idx: int) -> list[KITTILabel]:
        path = self.root / "label_2" / f"{self.samples[idx]}.txt"
        if not path.exists():
            return []
        labels = []
        with open(path) as f:
            for line in f:
                label = KITTILabel(line)
                if label.class_id >= 0:
                    labels.append(label)
        return labels

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.samples[idx]

        data = {"sample_id": sample_id}

        velodyne_path = self.root / "velodyne" / f"{sample_id}.bin"
        if velodyne_path.exists():
            data["point_cloud"] = torch.from_numpy(self.load_point_cloud(idx))

        image_path = self.root / "image_2" / f"{sample_id}.png"
        if image_path.exists():
            img = self.load_image(idx)
            data["image"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        calib_path = self.root / "calib" / f"{sample_id}.txt"
        if calib_path.exists():
            calib = self.load_calibration(idx)
            data["calibration"] = {
                "P2": torch.from_numpy(calib.P2).float(),
                "R0_rect": torch.from_numpy(calib.R0_rect).float(),
                "Tr_velo_to_cam": torch.from_numpy(calib.Tr_velo_to_cam).float(),
            }

        label_path = self.root / "label_2" / f"{sample_id}.txt"
        if label_path.exists():
            labels = self.load_labels(idx)
            if labels:
                data["boxes_3d"] = torch.tensor(
                    np.array([l.bbox_3d for l in labels]), dtype=torch.float32
                )
                data["labels"] = torch.tensor(
                    [l.class_id for l in labels], dtype=torch.long
                )

        return data
