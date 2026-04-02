"""Tests for 3D object detection modules."""

import torch
import numpy as np
import pytest


class TestDetectionHead:
    def test_forward_shape(self):
        from perception_stack.detection.heads import DetectionHead

        head = DetectionHead(in_channels=384, num_classes=3)
        x = torch.randn(2, 384, 25, 25)
        out = head(x)

        assert "cls_scores" in out
        assert "bbox_preds" in out
        assert "dir_preds" in out
        assert out["cls_scores"].shape == (2, 3, 25, 25)
        assert out["bbox_preds"].shape == (2, 7, 25, 25)
        assert out["dir_preds"].shape == (2, 2, 25, 25)

    def test_weight_init(self):
        from perception_stack.detection.heads import DetectionHead

        head = DetectionHead(in_channels=384, num_classes=3)
        # Classification bias should be initialized to -2.19 (focal loss)
        assert abs(head.cls_head.bias.data.mean().item() - (-2.19)) < 0.01


class TestNMS:
    def test_nms_basic(self):
        from perception_stack.detection.nms import nms_3d

        boxes = torch.tensor([
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [0.1, 0.1, 0.0, 2.0, 4.0, 1.5, 0.0],  # overlapping
            [20.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0],  # separate
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms_3d(boxes, scores, iou_threshold=0.5)
        # First and third should be kept (second overlaps with first)
        assert len(keep) == 2
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_nms_empty(self):
        from perception_stack.detection.nms import nms_3d

        boxes = torch.zeros(0, 7)
        scores = torch.zeros(0)
        keep = nms_3d(boxes, scores)
        assert len(keep) == 0

    def test_box_iou_bev(self):
        from perception_stack.detection.nms import box_iou_bev

        boxes_a = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]])
        boxes_b = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]])
        iou = box_iou_bev(boxes_a, boxes_b)
        assert abs(iou[0, 0].item() - 1.0) < 1e-5

    def test_box_iou_no_overlap(self):
        from perception_stack.detection.nms import box_iou_bev

        boxes_a = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        boxes_b = torch.tensor([[100.0, 100.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        iou = box_iou_bev(boxes_a, boxes_b)
        assert iou[0, 0].item() < 1e-5


class TestPillarFeatureNet:
    def test_forward(self):
        from perception_stack.detection.pointpillars import PillarFeatureNet

        net = PillarFeatureNet(in_channels=4, pillar_feat_channels=64, max_points_per_pillar=32)
        pillar_points = torch.randn(50, 32, 4)
        pillar_coords = torch.randint(0, 100, (50, 2))
        num_points = torch.randint(1, 32, (50,))

        out = net(pillar_points, pillar_coords, num_points)
        assert out.shape == (50, 64)


class TestPointPillarsBackbone:
    def test_forward(self):
        from perception_stack.detection.pointpillars import PointPillarsBackbone

        backbone = PointPillarsBackbone(in_channels=64)
        x = torch.randn(2, 64, 100, 100)
        out = backbone(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 384  # 128*3


class TestPointPillarsDetector:
    def test_init(self, detection_config):
        from perception_stack.detection.pointpillars import PointPillarsDetector

        detector = PointPillarsDetector(detection_config)
        assert detector.score_threshold == 0.3

    def test_forward(self, detection_config):
        from perception_stack.detection.pointpillars import PointPillarsDetector

        detector = PointPillarsDetector(detection_config)
        N_pillars = 100
        pillar_points = torch.randn(N_pillars, 32, 4)
        pillar_coords = torch.zeros(N_pillars, 3, dtype=torch.long)
        pillar_coords[:, 0] = 0  # batch idx
        pillar_coords[:, 1] = torch.randint(0, 100, (N_pillars,))
        pillar_coords[:, 2] = torch.randint(0, 100, (N_pillars,))
        num_points = torch.randint(1, 32, (N_pillars,))

        out = detector(pillar_points, pillar_coords, num_points, batch_size=1)
        assert "cls_scores" in out
        assert "bbox_preds" in out
