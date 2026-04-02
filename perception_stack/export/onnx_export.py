"""ONNX export utilities for deploying models to edge devices."""

import argparse
from pathlib import Path

import torch
import yaml


class PointPillarsExportWrapper(torch.nn.Module):
    """Wrapper to export the PointPillars backbone + head to ONNX.

    Takes pseudo-image (BEV features) as input since pillar creation
    is typically handled in pre-processing on the device.
    """

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pseudo_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(pseudo_image)
        preds = self.head(features)
        return preds["cls_scores"], preds["bbox_preds"], preds["dir_preds"]


class LaneDetectorExportWrapper(torch.nn.Module):
    """Wrapper for lane segmentation model export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(image)
        return out["segmentation"], out["embedding"]


def export_detector(config: dict, output_dir: str | None = None) -> str:
    """Export PointPillars detector to ONNX.

    Args:
        config: full config dict
        output_dir: output directory (overrides config)

    Returns:
        Path to exported ONNX file
    """
    from perception_stack.detection.pointpillars import PointPillarsBackbone
    from perception_stack.detection.heads import DetectionHead

    det_config = config.get("detection", {})
    export_config = config.get("export", {})

    pillar_features = det_config.get("pillar_features", 64)
    num_classes = det_config.get("num_classes", 3)
    grid_size = det_config.get("grid_size", [500, 500])
    opset = export_config.get("opset_version", 17)
    out_dir = output_dir or export_config.get("output_dir", "./outputs/onnx")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    backbone = PointPillarsBackbone(pillar_features)
    head = DetectionHead(in_channels=384, num_classes=num_classes)
    wrapper = PointPillarsExportWrapper(backbone, head)
    wrapper.eval()

    # Dummy input: pseudo-image
    dummy_input = torch.randn(1, pillar_features, grid_size[0] // 2, grid_size[1] // 2)

    output_path = str(Path(out_dir) / "pointpillars.onnx")

    dynamic_axes = None
    if export_config.get("dynamic_axes", True):
        dynamic_axes = {
            "pseudo_image": {0: "batch_size"},
            "cls_scores": {0: "batch_size"},
            "bbox_preds": {0: "batch_size"},
            "dir_preds": {0: "batch_size"},
        }

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["pseudo_image"],
        output_names=["cls_scores", "bbox_preds", "dir_preds"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported PointPillars to {output_path}")
    return output_path


def export_lane_detector(config: dict, output_dir: str | None = None) -> str:
    """Export lane segmentation model to ONNX.

    Args:
        config: full config dict
        output_dir: output directory

    Returns:
        Path to exported ONNX file
    """
    from perception_stack.lanes.segmentation import LaneSegmentationNet

    lane_config = config.get("lane_detection", {})
    export_config = config.get("export", {})

    num_classes = lane_config.get("num_classes", 3)
    opset = export_config.get("opset_version", 17)
    out_dir = output_dir or export_config.get("output_dir", "./outputs/onnx")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model = LaneSegmentationNet(num_classes=num_classes)
    wrapper = LaneDetectorExportWrapper(model)
    wrapper.eval()

    H = lane_config.get("image_height", 720)
    W = lane_config.get("image_width", 1280)
    dummy_input = torch.randn(1, 3, H, W)

    output_path = str(Path(out_dir) / "lane_detector.onnx")

    dynamic_axes = None
    if export_config.get("dynamic_axes", True):
        dynamic_axes = {
            "image": {0: "batch_size"},
            "segmentation": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        }

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["segmentation", "embedding"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported lane detector to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--model", choices=["detector", "lane", "all"], default="all"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.model in ("detector", "all"):
        export_detector(config, args.output)
    if args.model in ("lane", "all"):
        export_lane_detector(config, args.output)


if __name__ == "__main__":
    main()
