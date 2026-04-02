"""Tests for ROS2 message types and node stubs."""

import numpy as np
import pytest


class TestMessages:
    def test_bounding_box_3d(self):
        from perception_stack.ros2.messages import BoundingBox3D, Point3D, Pose, Vector3

        bbox = BoundingBox3D(
            center=Pose(position=Point3D(x=10.0, y=5.0, z=0.0)),
            size=Vector3(x=2.0, y=4.5, z=1.5),
            class_id=0,
            class_name="vehicle",
            score=0.95,
        )
        d = bbox.to_dict()
        assert d["center"]["position"]["x"] == 10.0
        assert d["class_name"] == "vehicle"

    def test_detection_3d_array(self):
        from perception_stack.ros2.messages import Detection3DArray, BoundingBox3D

        arr = Detection3DArray(detections=[BoundingBox3D(), BoundingBox3D()])
        d = arr.to_dict()
        assert len(d["detections"]) == 2

    def test_track_array(self):
        from perception_stack.ros2.messages import TrackArray, TrackState

        arr = TrackArray(tracks=[TrackState(track_id=1), TrackState(track_id=2)])
        d = arr.to_dict()
        assert len(d["tracks"]) == 2
        assert d["tracks"][0]["track_id"] == 1

    def test_lane_array(self):
        from perception_stack.ros2.messages import LaneArray, Lane, LanePoint

        lane = Lane(lane_id=0, points=[LanePoint(x=1.0, y=2.0)], confidence=0.9)
        arr = LaneArray(lanes=[lane])
        d = arr.to_dict()
        assert len(d["lanes"]) == 1
        assert d["lanes"][0]["confidence"] == 0.9

    def test_occupancy_grid_msg(self):
        from perception_stack.ros2.messages import OccupancyGridMsg

        msg = OccupancyGridMsg(width=10, height=10, resolution=1.0, data=[0.5] * 100)
        d = msg.to_dict()
        assert d["width"] == 10
        assert len(d["data"]) == 100


class TestPerceptionNodeStub:
    def test_init(self):
        from perception_stack.ros2.nodes import PerceptionNodeStub

        node = PerceptionNodeStub()
        assert node.node_name == "perception_node"

    def test_publish_tracks(self):
        from perception_stack.ros2.nodes import PerceptionNodeStub
        from perception_stack.tracking.tracker import MultiObjectTracker

        tracker = MultiObjectTracker()
        det = np.array([[10.0, 5.0, 0.0, 2.0, 4.5, 1.5, 0.0]])
        for _ in range(3):
            tracker.update(det)

        node = PerceptionNodeStub()
        msg = node.publish_tracks(tracker.tracks)
        assert len(msg.tracks) >= 1

    def test_publish_occupancy(self):
        from perception_stack.ros2.nodes import PerceptionNodeStub
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid(width=10, height=10)
        node = PerceptionNodeStub()
        msg = node.publish_occupancy(grid)
        assert msg.width == 10
        assert len(msg.data) == 100
