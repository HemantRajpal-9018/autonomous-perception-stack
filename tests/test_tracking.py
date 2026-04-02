"""Tests for multi-object tracking modules."""

import numpy as np
import pytest


class TestKalmanFilter3D:
    def test_initialize(self):
        from perception_stack.tracking.kalman import KalmanFilter3D

        kf = KalmanFilter3D()
        measurement = np.array([10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.1])
        kf.initialize(measurement)

        assert abs(kf.position[0] - 10.0) < 1e-5
        assert abs(kf.position[1] - 5.0) < 1e-5
        assert abs(kf.yaw - 0.1) < 1e-5

    def test_predict(self):
        from perception_stack.tracking.kalman import KalmanFilter3D

        kf = KalmanFilter3D(dt=0.1)
        kf.initialize(np.array([0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]))
        # Set velocity
        kf.x[3] = 10.0  # vx = 10 m/s

        state = kf.predict()
        # x should advance by vx * dt = 1.0
        assert abs(state[0] - 1.0) < 1e-5

    def test_update(self):
        from perception_stack.tracking.kalman import KalmanFilter3D

        kf = KalmanFilter3D()
        kf.initialize(np.array([0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]))
        kf.predict()

        measurement = np.array([0.5, 0.3, 0.0, 2.0, 4.0, 1.5, 0.0])
        state = kf.update(measurement)

        # After update, state should be closer to measurement
        assert abs(state[0]) < 1.0
        assert abs(state[1]) < 1.0

    def test_bbox_3d_property(self):
        from perception_stack.tracking.kalman import KalmanFilter3D

        kf = KalmanFilter3D()
        kf.initialize(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]))
        bbox = kf.bbox_3d
        assert len(bbox) == 7
        assert abs(bbox[0] - 1.0) < 1e-5
        assert abs(bbox[6] - 0.5) < 1e-5


class TestHungarianAssignment:
    def test_basic_assignment(self):
        from perception_stack.tracking.hungarian import hungarian_assignment

        cost = np.array([
            [0.1, 0.9, 0.9],
            [0.9, 0.2, 0.9],
            [0.9, 0.9, 0.3],
        ])
        matches, unmatched_t, unmatched_d = hungarian_assignment(cost, threshold=0.5)
        assert len(matches) == 3
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_threshold_gating(self):
        from perception_stack.tracking.hungarian import hungarian_assignment

        cost = np.array([
            [0.1, 0.9],
            [0.9, 0.9],
        ])
        matches, unmatched_t, unmatched_d = hungarian_assignment(cost, threshold=0.5)
        assert len(matches) == 1
        assert 1 in unmatched_t

    def test_empty_cost(self):
        from perception_stack.tracking.hungarian import hungarian_assignment

        cost = np.zeros((0, 3))
        matches, unmatched_t, unmatched_d = hungarian_assignment(cost)
        assert len(matches) == 0
        assert len(unmatched_d) == 3

    def test_iou_3d(self):
        from perception_stack.tracking.hungarian import iou_3d_axis_aligned

        boxes_a = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        boxes_b = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        iou = iou_3d_axis_aligned(boxes_a, boxes_b)
        assert abs(iou[0, 0] - 1.0) < 1e-5


class TestMultiObjectTracker:
    def test_create_and_track(self, tracking_config):
        from perception_stack.tracking.tracker import MultiObjectTracker

        tracker = MultiObjectTracker(tracking_config)
        detections = np.array([
            [10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.0],
            [-5.0, 3.0, 0.0, 0.6, 0.6, 1.8, 0.0],
        ])
        labels = np.array([0, 1])

        tracks = tracker.update(detections, labels)
        assert len(tracks) >= 2

    def test_track_persistence(self, tracking_config):
        from perception_stack.tracking.tracker import MultiObjectTracker

        tracker = MultiObjectTracker(tracking_config)
        det = np.array([[10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.0]])

        # First 3 updates to confirm track
        for _ in range(3):
            tracker.update(det)

        # Track should be confirmed
        confirmed = [t for t in tracker.tracks if t.is_confirmed]
        assert len(confirmed) >= 1

    def test_track_deletion(self):
        from perception_stack.tracking.tracker import MultiObjectTracker

        tracker = MultiObjectTracker({"max_age": 2})
        det = np.array([[10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.0]])
        tracker.update(det)

        # Send empty detections for max_age + 1 frames
        for _ in range(5):
            tracker.update(np.zeros((0, 7)))

        # Track should be deleted
        assert len(tracker.tracks) == 0

    def test_reset(self, tracking_config):
        from perception_stack.tracking.tracker import MultiObjectTracker

        tracker = MultiObjectTracker(tracking_config)
        det = np.array([[10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.0]])
        tracker.update(det)
        assert len(tracker.tracks) > 0

        tracker.reset()
        assert len(tracker.tracks) == 0
