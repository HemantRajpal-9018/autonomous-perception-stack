"""ROS2-compatible message type definitions.

These are Python dataclass equivalents of the ROS2 message types used in
the perception pipeline. They can be serialized to/from ROS2 messages
when rclpy is available.
"""

from dataclasses import dataclass, field
import time


@dataclass
class Header:
    """Standard message header."""
    stamp: float = 0.0
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        if self.stamp == 0.0:
            self.stamp = time.time()

    def to_dict(self) -> dict:
        return {"stamp": self.stamp, "frame_id": self.frame_id}


@dataclass
class Point3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Point3D = field(default_factory=Point3D)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class BoundingBox3D:
    """3D bounding box message."""
    header: Header = field(default_factory=Header)
    center: Pose = field(default_factory=Pose)
    size: Vector3 = field(default_factory=Vector3)
    class_id: int = 0
    class_name: str = ""
    score: float = 0.0
    track_id: int = -1

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "center": {
                "position": {"x": self.center.position.x,
                              "y": self.center.position.y,
                              "z": self.center.position.z},
                "orientation": {"x": self.center.orientation.x,
                                 "y": self.center.orientation.y,
                                 "z": self.center.orientation.z,
                                 "w": self.center.orientation.w},
            },
            "size": {"x": self.size.x, "y": self.size.y, "z": self.size.z},
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "track_id": self.track_id,
        }


@dataclass
class Detection3DArray:
    """Array of 3D detections."""
    header: Header = field(default_factory=Header)
    detections: list[BoundingBox3D] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class TrackState:
    """State of a tracked object."""
    track_id: int = 0
    bbox: BoundingBox3D = field(default_factory=BoundingBox3D)
    velocity: Vector3 = field(default_factory=Vector3)
    age: int = 0
    hits: int = 0
    is_confirmed: bool = False


@dataclass
class TrackArray:
    """Array of tracked objects."""
    header: Header = field(default_factory=Header)
    tracks: list[TrackState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "bbox": t.bbox.to_dict(),
                    "velocity": {"x": t.velocity.x, "y": t.velocity.y, "z": t.velocity.z},
                    "age": t.age,
                    "hits": t.hits,
                    "is_confirmed": t.is_confirmed,
                }
                for t in self.tracks
            ],
        }


@dataclass
class LanePoint:
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0


@dataclass
class Lane:
    lane_id: int = 0
    points: list[LanePoint] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    lane_type: str = "solid"  # solid, dashed, double
    confidence: float = 0.0


@dataclass
class LaneArray:
    """Array of detected lanes."""
    header: Header = field(default_factory=Header)
    lanes: list[Lane] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "lanes": [
                {
                    "lane_id": l.lane_id,
                    "points": [{"x": p.x, "y": p.y, "confidence": p.confidence}
                               for p in l.points],
                    "coefficients": l.coefficients,
                    "lane_type": l.lane_type,
                    "confidence": l.confidence,
                }
                for l in self.lanes
            ],
        }


@dataclass
class OccupancyGridMsg:
    """Occupancy grid message."""
    header: Header = field(default_factory=Header)
    width: int = 200
    height: int = 200
    resolution: float = 0.5
    origin_x: float = -50.0
    origin_y: float = -50.0
    data: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": {"x": self.origin_x, "y": self.origin_y},
            "data": self.data,
        }
