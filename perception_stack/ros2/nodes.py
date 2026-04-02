"""ROS2 node stubs for perception pipeline integration.

These stubs define the node structure and topic interfaces.
When rclpy is available, they can be extended to full ROS2 nodes.
"""

from perception_stack.ros2.messages import (
    BoundingBox3D,
    Detection3DArray,
    Header,
    LaneArray,
    OccupancyGridMsg,
    Point3D,
    Pose,
    Quaternion,
    TrackArray,
    TrackState,
    Vector3,
)

import numpy as np


class PerceptionNodeStub:
    """Stub for the main perception ROS2 node.

    Topics:
      Subscriptions:
        /lidar/points         — sensor_msgs/PointCloud2
        /camera/*/image_raw   — sensor_msgs/Image
        /camera/*/camera_info — sensor_msgs/CameraInfo

      Publishers:
        /perception/detections   — Detection3DArray
        /perception/tracks       — TrackArray
        /perception/lanes        — LaneArray
        /perception/occupancy    — OccupancyGridMsg
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.node_name = "perception_node"
        self.frame_count = 0

    def on_lidar_callback(self, point_cloud: np.ndarray) -> Detection3DArray:
        """Process LiDAR point cloud and return detections.

        This is a stub. In a real ROS2 node, this would be a subscriber callback.
        """
        from perception_stack.detection.pointpillars import PointPillarsDetector

        self.frame_count += 1
        header = Header(frame_id="lidar")

        # Placeholder: in production, this runs the full pipeline
        return Detection3DArray(header=header, detections=[])

    def on_camera_callback(self, image: np.ndarray, camera_name: str) -> LaneArray:
        """Process camera image for lane detection.

        This is a stub. In a real ROS2 node, this would be a subscriber callback.
        """
        header = Header(frame_id=camera_name)
        return LaneArray(header=header, lanes=[])

    def publish_tracks(self, tracks: list) -> TrackArray:
        """Convert tracker output to ROS2 message."""
        header = Header(frame_id="base_link")
        track_states = []

        for track in tracks:
            bbox = track.bbox_3d
            ts = TrackState(
                track_id=track.track_id,
                bbox=BoundingBox3D(
                    center=Pose(
                        position=Point3D(x=float(bbox[0]), y=float(bbox[1]), z=float(bbox[2])),
                        orientation=Quaternion(),
                    ),
                    size=Vector3(x=float(bbox[3]), y=float(bbox[4]), z=float(bbox[5])),
                    class_id=track.label,
                    score=1.0,
                    track_id=track.track_id,
                ),
                velocity=Vector3(
                    x=float(track.velocity[0]),
                    y=float(track.velocity[1]),
                    z=float(track.velocity[2]),
                ),
                age=track.age,
                hits=track.hits,
                is_confirmed=track.is_confirmed,
            )
            track_states.append(ts)

        return TrackArray(header=header, tracks=track_states)

    def publish_occupancy(self, grid) -> OccupancyGridMsg:
        """Convert occupancy grid to ROS2 message."""
        prob = grid.probability_grid.flatten().tolist()
        return OccupancyGridMsg(
            header=Header(frame_id="base_link"),
            width=grid.width,
            height=grid.height,
            resolution=grid.resolution,
            origin_x=grid.origin[0],
            origin_y=grid.origin[1],
            data=prob,
        )
