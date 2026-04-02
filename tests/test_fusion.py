"""Tests for sensor fusion modules."""

import numpy as np
import pytest


class TestProjectionMatrix:
    def test_project_lidar_to_image(self, sample_intrinsic, sample_extrinsic):
        from perception_stack.fusion.projection import ProjectionMatrix

        proj = ProjectionMatrix(sample_intrinsic, sample_extrinsic)
        points = np.array([[5.0, 0.0, 1.0], [10.0, 2.0, 0.5], [-1.0, 0.0, 1.0]])

        uv, depth = proj.project_lidar_to_image(points)
        assert uv.ndim == 2
        assert depth.ndim == 1

    def test_unproject(self, sample_intrinsic, sample_extrinsic):
        from perception_stack.fusion.projection import ProjectionMatrix

        proj = ProjectionMatrix(sample_intrinsic, sample_extrinsic)
        uv = np.array([[609.5, 172.8]])  # principal point
        depth = np.array([10.0])

        world_pts = proj.unproject_image_to_world(uv, depth)
        assert world_pts.shape == (1, 3)
        # At principal point with identity extrinsic, should project to (0, 0, 10)
        assert abs(world_pts[0, 2] - 10.0) < 0.1

    def test_roundtrip(self, sample_intrinsic, sample_extrinsic):
        from perception_stack.fusion.projection import ProjectionMatrix

        proj = ProjectionMatrix(sample_intrinsic, sample_extrinsic)
        original = np.array([[5.0, 3.0, 10.0]])

        uv, depth = proj.project_world_to_image(original)
        if len(uv) > 0:
            recovered = proj.unproject_image_to_world(uv, depth)
            np.testing.assert_allclose(recovered, original, atol=0.1)


class TestCameraLiDARFusion:
    def test_associate(self):
        from perception_stack.fusion.camera_lidar import CameraLiDARFusion

        fusion = CameraLiDARFusion(association_threshold=2.0)
        cam_boxes = np.array([
            [10.0, 5.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [20.0, 10.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        ])
        lid_boxes = np.array([
            [10.2, 5.1, 0.0, 2.0, 4.0, 1.5, 0.0],
            [30.0, 30.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        ])

        matches, unmatched_cam, unmatched_lid = fusion.associate_detections(cam_boxes, lid_boxes)
        assert len(matches) >= 1

    def test_fuse_boxes(self):
        from perception_stack.fusion.camera_lidar import CameraLiDARFusion

        fusion = CameraLiDARFusion()
        cam_boxes = np.array([[10.0, 5.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
        cam_scores = np.array([0.8])
        lid_boxes = np.array([[10.1, 5.1, 0.0, 2.1, 4.1, 1.5, 0.0]])
        lid_scores = np.array([0.9])

        fused_boxes, fused_scores = fusion.fuse_boxes(
            cam_boxes, cam_scores, lid_boxes, lid_scores
        )
        assert len(fused_boxes) >= 1
        assert len(fused_scores) >= 1

    def test_fuse_empty(self):
        from perception_stack.fusion.camera_lidar import CameraLiDARFusion

        fusion = CameraLiDARFusion()
        boxes, scores = fusion.fuse_boxes(
            np.zeros((0, 7)), np.zeros(0),
            np.zeros((0, 7)), np.zeros(0),
        )
        assert len(boxes) == 0
