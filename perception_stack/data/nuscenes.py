"""nuScenes dataset loader for 3D object detection.

Supports reading:
  - LiDAR point clouds (.bin)
  - Multi-camera images (.jpg)
  - Calibration from JSON scene metadata
  - 3D annotations
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


NUSCENES_CLASSES = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "trailer": 0,
    "construction_vehicle": 0,
    "pedestrian": 1,
    "motorcycle": 2,
    "bicycle": 2,
    "barrier": -1,
    "traffic_cone": -1,
}

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesDataset(Dataset):
    """nuScenes dataset loader.

    Simplified loader that reads from pre-processed nuScenes directory:
        root/
            samples/
                LIDAR_TOP/       — <token>.bin
                CAM_FRONT/       — <token>.jpg
                CAM_FRONT_RIGHT/ — <token>.jpg
                ...
            v1.0-trainval/
                sample.json
                sample_data.json
                sample_annotation.json
                ego_pose.json
                calibrated_sensor.json
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
        self.samples: list[dict] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        meta_dir = self.root / "v1.0-trainval"
        if not meta_dir.exists():
            return

        sample_file = meta_dir / "sample.json"
        if sample_file.exists():
            with open(sample_file) as f:
                self.samples = json.load(f)

        # Load auxiliary tables
        self._sample_data = {}
        sd_file = meta_dir / "sample_data.json"
        if sd_file.exists():
            with open(sd_file) as f:
                for sd in json.load(f):
                    self._sample_data[sd["token"]] = sd

        self._annotations = {}
        ann_file = meta_dir / "sample_annotation.json"
        if ann_file.exists():
            with open(ann_file) as f:
                for ann in json.load(f):
                    sample_token = ann.get("sample_token", "")
                    if sample_token not in self._annotations:
                        self._annotations[sample_token] = []
                    self._annotations[sample_token].append(ann)

        self._calibrated_sensors = {}
        cs_file = meta_dir / "calibrated_sensor.json"
        if cs_file.exists():
            with open(cs_file) as f:
                for cs in json.load(f):
                    self._calibrated_sensors[cs["token"]] = cs

    def __len__(self) -> int:
        return len(self.samples)

    def _load_point_cloud(self, path: str | Path) -> np.ndarray:
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        return points

    def _load_image(self, path: str | Path) -> np.ndarray:
        return np.array(Image.open(path))

    def _parse_annotation(self, ann: dict) -> dict | None:
        category = ann.get("category_name", "").split(".")[-1]
        class_id = NUSCENES_CLASSES.get(category, -1)
        if class_id < 0:
            return None

        translation = np.array(ann["translation"])
        size = np.array(ann["size"])  # wlh
        rotation = ann.get("rotation", [1, 0, 0, 0])
        # Convert quaternion to yaw (simplified: extract z-rotation)
        w, x, y, z = rotation
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        return {
            "class_id": class_id,
            "bbox_3d": np.array([
                translation[0], translation[1], translation[2],
                size[0], size[1], size[2], yaw,
            ]),
        }

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        token = sample["token"]
        data: dict = {"sample_token": token}

        # Load LiDAR
        lidar_path = self.root / "samples" / "LIDAR_TOP" / f"{token}.bin"
        if lidar_path.exists():
            pc = self._load_point_cloud(lidar_path)
            data["point_cloud"] = torch.from_numpy(pc[:, :4])  # x, y, z, intensity

        # Load camera images
        images = []
        for cam_name in CAMERA_NAMES:
            cam_path = self.root / "samples" / cam_name / f"{token}.jpg"
            if cam_path.exists():
                img = self._load_image(cam_path)
                images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        if images:
            # Pad/crop to same size if needed
            data["images"] = torch.stack(images)

        # Load annotations
        anns = self._annotations.get(token, [])
        parsed = [self._parse_annotation(a) for a in anns]
        parsed = [p for p in parsed if p is not None]
        if parsed:
            data["boxes_3d"] = torch.tensor(
                np.array([p["bbox_3d"] for p in parsed]), dtype=torch.float32
            )
            data["labels"] = torch.tensor(
                [p["class_id"] for p in parsed], dtype=torch.long
            )

        return data
