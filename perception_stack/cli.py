"""Command-line interface for the perception stack."""

import argparse
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Perception Stack")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument(
        "--mode",
        choices=["detect", "track", "export", "visualize"],
        default="detect",
        help="Pipeline mode",
    )
    parser.add_argument("--input", type=str, help="Input data path")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Perception Stack v{__import__('perception_stack').__version__}")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")

    if args.mode == "detect":
        from perception_stack.detection.pointpillars import PointPillarsDetector

        detector = PointPillarsDetector(config["detection"])
        print("PointPillars detector initialized.")

    elif args.mode == "track":
        from perception_stack.tracking.tracker import MultiObjectTracker

        tracker = MultiObjectTracker(config["tracking"])
        print("Multi-object tracker initialized.")

    elif args.mode == "export":
        from perception_stack.export.onnx_export import export_detector

        export_detector(config)
        print("ONNX export complete.")

    elif args.mode == "visualize":
        print("Visualization mode — use perception_stack.visualization module directly.")

    return


if __name__ == "__main__":
    main()
