# Autonomous Perception Stack

End-to-end perception pipeline for autonomous driving — from raw sensor data to structured environment representation for motion planning.

## Architecture

```
                        ┌──────────────────────────────────────────────────────┐
                        │              AUTONOMOUS PERCEPTION STACK             │
                        └──────────────────────────────────────────────────────┘

  ┌─────────────┐   ┌─────────────┐
  │  Camera x6  │   │  LiDAR x1   │
  │  (1600x900) │   │ (100k pts)  │
  └──────┬──────┘   └──────┬──────┘
         │                 │
         ▼                 ▼
  ┌─────────────┐   ┌─────────────┐
  │  BEV Trans  │   │   Voxelize  │
  │ Homography/ │   │  (Pillars)  │
  │ Lift-Splat  │   │             │
  └──────┬──────┘   └──────┬──────┘
         │                 │
         ▼                 ▼
  ┌─────────────┐   ┌──────────────┐
  │    Lane     │   │ PointPillars │
  │  Detection  │   │  3D Detect   │
  │  Seg + Poly │   │  Backbone +  │
  └──────┬──────┘   │  Head + NMS  │
         │          └──────┬───────┘
         │                 │
         │    ┌────────────┴────────────┐
         │    │     Sensor Fusion       │
         │    │  Camera + LiDAR Late    │
         │    │  Fusion (Weighted Avg)  │
         │    └────────────┬────────────┘
         │                 │
         │    ┌────────────┴────────────┐
         │    │   Multi-Object Tracker  │
         │    │  Kalman Filter State    │
         │    │  + Hungarian Matching   │
         │    └────────────┬────────────┘
         │                 │
         ▼                 ▼
  ┌──────────────────────────────────┐
  │        Occupancy Grid            │
  │  Probabilistic BEV Grid (200x200)│
  │  Log-odds Bayesian Updates       │
  │  Bresenham Ray Tracing           │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │        Motion Planner            │
  │     (downstream consumer)        │
  └──────────────────────────────────┘
```

## Modules

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `bev/` | Bird's Eye View transforms | `HomographyBEV`, `LiftSplatShoot` |
| `detection/` | 3D object detection | `PointPillarsDetector`, `DetectionHead`, `nms_3d` |
| `lanes/` | Lane detection | `LaneSegmentationNet`, `PolynomialLaneFitter` |
| `tracking/` | Multi-object tracking | `MultiObjectTracker`, `KalmanFilter3D`, `hungarian_assignment` |
| `occupancy/` | Occupancy grid mapping | `OccupancyGrid` |
| `fusion/` | Sensor fusion | `CameraLiDARFusion`, `ProjectionMatrix` |
| `data/` | Dataset loaders | `KITTIDataset`, `NuScenesDataset` |
| `visualization/` | Viz tools | `draw_3d_boxes`, `plot_bev`, `plot_tracking`, `render_point_cloud_bev` |
| `export/` | ONNX export | `export_detector`, `export_lane_detector` |
| `ros2/` | ROS2 interfaces | `Detection3DArray`, `TrackArray`, `OccupancyGridMsg`, `PerceptionNodeStub` |

## Installation

```bash
# Clone
git clone <repo-url>
cd autonomous-perception-stack

# Install
pip install -e .

# Install with dev tools
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy, SciPy, OpenCV, Matplotlib
- ONNX + ONNXRuntime (for export)

## Quick Start

### 3D Object Detection

```python
import torch
from perception_stack.detection import PointPillarsDetector

config = {
    "num_classes": 3,
    "pillar_features": 64,
    "grid_size": [500, 500],
    "score_threshold": 0.3,
    "nms_threshold": 0.5,
}

detector = PointPillarsDetector(config)

# Prepare pillarized point cloud
pillar_points = torch.randn(200, 32, 4)       # (N_pillars, max_points, 4)
pillar_coords = torch.randint(0, 500, (200, 3)) # (N_pillars, 3) batch,y,x
num_pts = torch.randint(1, 32, (200,))

results = detector.predict(pillar_points, pillar_coords, num_pts, batch_size=1)
# results[0] = {'boxes': (K, 7), 'scores': (K,), 'labels': (K,)}
```

### Multi-Object Tracking

```python
import numpy as np
from perception_stack.tracking import MultiObjectTracker

tracker = MultiObjectTracker({
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
})

# Each frame: pass detections [x, y, z, w, l, h, yaw]
detections = np.array([
    [10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.1],
    [-5.0, 3.0, 0.0, 0.6, 0.6, 1.8, 0.0],
])
labels = np.array([0, 1])  # vehicle, pedestrian

tracks = tracker.update(detections, labels)
for t in tracks:
    print(f"Track {t.track_id}: pos={t.bbox_3d[:3]}, vel={t.velocity}")
```

### Lane Detection

```python
import torch
from perception_stack.lanes import LaneSegmentationNet, PolynomialLaneFitter

model = LaneSegmentationNet(num_classes=3)
image = torch.randn(1, 3, 720, 1280)
output = model(image)
seg_mask = output["segmentation"].argmax(dim=1)[0]

fitter = PolynomialLaneFitter(degree=3)
lanes = fitter.fit_lanes_torch(seg_mask)
```

### BEV Transform

```python
import torch
from perception_stack.bev import HomographyBEV

bev = HomographyBEV(image_size=(900, 1600), bev_size=(200, 200), num_cameras=6)
images = torch.randn(1, 6, 3, 900, 1600)
intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 6, -1, -1)
extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 6, -1, -1)

bev_features = bev(images, intrinsics, extrinsics)  # (1, 64, 200, 200)
```

### Sensor Fusion

```python
import numpy as np
from perception_stack.fusion import CameraLiDARFusion

fusion = CameraLiDARFusion(camera_weight=0.4, lidar_weight=0.6)
fused_boxes, fused_scores = fusion.fuse_boxes(
    camera_boxes, camera_scores, lidar_boxes, lidar_scores
)
```

### Occupancy Grid

```python
from perception_stack.occupancy import OccupancyGrid

grid = OccupancyGrid(width=200, height=200, resolution=0.5)
grid.update_from_detections(tracked_boxes)
grid.update_from_lidar(point_cloud, sensor_position)
costmap = grid.get_costmap()  # (200, 200) float [0, 1]
```

### Visualization

```python
from perception_stack.visualization import plot_bev, draw_3d_boxes_on_image

# BEV plot with boxes and point cloud
fig = plot_bev(boxes=detections, labels=labels, point_cloud=lidar_points)
fig.savefig("bev.png")

# 3D boxes on camera image
img = draw_3d_boxes_on_image(camera_image, boxes_3d, labels, projection_matrix)
```

## Dataset Setup

### KITTI

```bash
# Download from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
data/kitti/
├── velodyne/        # xxxxxx.bin
├── image_2/         # xxxxxx.png
├── calib/           # xxxxxx.txt
└── label_2/         # xxxxxx.txt
```

```python
from perception_stack.data import KITTIDataset
dataset = KITTIDataset("data/kitti", split="train")
sample = dataset[0]  # {'point_cloud', 'image', 'calibration', 'boxes_3d', 'labels'}
```

### nuScenes

```bash
# Download from https://www.nuscenes.org/nuscenes
data/nuscenes/
├── samples/
│   ├── LIDAR_TOP/
│   ├── CAM_FRONT/
│   └── ...
└── v1.0-trainval/
    ├── sample.json
    ├── sample_data.json
    └── sample_annotation.json
```

```python
from perception_stack.data import NuScenesDataset
dataset = NuScenesDataset("data/nuscenes", split="train")
```

## ONNX Export

```bash
# Export all models
make export-onnx

# Or via Python
python -m perception_stack.export.onnx_export --config configs/default.yaml --model all
```

Exports:
- `pointpillars.onnx` — 3D detector backbone + head
- `lane_detector.onnx` — Lane segmentation network

## Benchmarks

| Model | Input | Parameters | FLOPs (est.) | Latency* |
|-------|-------|-----------|--------------|----------|
| PointPillars | 20k pillars | ~4.8M | ~15 GFLOPs | ~25 ms |
| Lane Seg Net | 720x1280 | ~0.8M | ~8 GFLOPs | ~12 ms |
| Homography BEV | 6x 900x1600 | ~0.5M | ~6 GFLOPs | ~18 ms |
| Tracking (Kalman + Hungarian) | N detections | N/A | O(N^3) | ~1 ms |
| Occupancy Grid | 200x200 | N/A | O(N) | ~5 ms |

*Estimated on NVIDIA RTX 3090. Actual performance varies by hardware.

## ROS2 Integration

Message types are defined in `perception_stack.ros2.messages`:

| Message | Topic | Description |
|---------|-------|-------------|
| `Detection3DArray` | `/perception/detections` | 3D bounding box detections |
| `TrackArray` | `/perception/tracks` | Tracked objects with velocity |
| `LaneArray` | `/perception/lanes` | Detected lane polynomials |
| `OccupancyGridMsg` | `/perception/occupancy` | Probabilistic occupancy grid |

Node stub: `PerceptionNodeStub` in `perception_stack.ros2.nodes`.

## Docker

```bash
# Build and run
docker compose build
docker compose up

# Export models only
docker compose run --rm tensorrt-export
```

## Development

```bash
make dev        # Install dev dependencies
make test       # Run tests
make test-cov   # Run tests with coverage
make lint       # Run linters
make format     # Auto-format code
make clean      # Remove build artifacts
```

## Testing

64 tests covering all modules:

```bash
pytest tests/ -v
```

## Project Structure

```
perception_stack/
├── __init__.py
├── cli.py                    # CLI entry point
├── bev/
│   ├── homography.py         # Homography-based BEV projection
│   └── lift_splat.py         # Lift-Splat-Shoot BEV encoder
├── detection/
│   ├── pointpillars.py       # PointPillars 3D detector
│   ├── heads.py              # SSD-style detection head
│   └── nms.py                # 3D Non-Maximum Suppression
├── lanes/
│   ├── segmentation.py       # U-Net lane segmentation
│   └── polynomial.py         # Polynomial curve fitting
├── tracking/
│   ├── kalman.py             # 3D Kalman filter
│   ├── hungarian.py          # Hungarian assignment + 3D IoU
│   └── tracker.py            # Multi-object tracker
├── occupancy/
│   └── grid.py               # Probabilistic occupancy grid
├── fusion/
│   ├── camera_lidar.py       # Camera-LiDAR late fusion
│   └── projection.py         # Projection matrix utilities
├── data/
│   ├── kitti.py              # KITTI dataset loader
│   └── nuscenes.py           # nuScenes dataset loader
├── visualization/
│   ├── boxes_3d.py           # 3D bounding box drawing
│   ├── bev_plot.py           # BEV visualization
│   ├── tracking_viz.py       # Tracking trajectory visualization
│   └── point_cloud_viz.py    # Point cloud rendering
├── export/
│   └── onnx_export.py        # ONNX model export
└── ros2/
    ├── messages.py            # ROS2 message types
    └── nodes.py               # ROS2 node stubs
```

## License

MIT
