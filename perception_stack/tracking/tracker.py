"""Multi-object tracker with Kalman filtering and Hungarian assignment."""

from dataclasses import dataclass, field

import numpy as np

from perception_stack.tracking.kalman import KalmanFilter3D
from perception_stack.tracking.hungarian import iou_3d_axis_aligned, hungarian_assignment


@dataclass
class Track:
    """A tracked object."""

    track_id: int
    kf: KalmanFilter3D
    label: int = 0
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def bbox_3d(self) -> np.ndarray:
        return self.kf.bbox_3d

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity


class MultiObjectTracker:
    """Multi-object tracker using Kalman filter + Hungarian algorithm.

    Lifecycle:
      1. Predict all tracks
      2. Associate detections to tracks via IoU + Hungarian
      3. Update matched tracks
      4. Create new tracks for unmatched detections
      5. Delete stale tracks
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_age = config.get("max_age", 30)
        self.min_hits = config.get("min_hits", 3)
        self.iou_threshold = config.get("iou_threshold", 0.3)
        self.dt = config.get("dt", 0.1)
        self.process_noise = config.get("process_noise", 1.0)
        self.measurement_noise = config.get("measurement_noise", 0.5)

        self.tracks: list[Track] = []
        self._next_id = 0

    def _create_track(self, detection: np.ndarray, label: int = 0) -> Track:
        kf = KalmanFilter3D(
            dt=self.dt,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        kf.initialize(detection)
        track = Track(track_id=self._next_id, kf=kf, label=label)
        self._next_id += 1
        return track

    def update(
        self, detections: np.ndarray, labels: np.ndarray | None = None
    ) -> list[Track]:
        """Run one tracking step.

        Args:
            detections: (N, 7) detected bounding boxes [x, y, z, w, l, h, yaw]
            labels: (N,) class labels (optional)

        Returns:
            List of active (confirmed) tracks
        """
        if labels is None:
            labels = np.zeros(len(detections), dtype=int)

        # 1. Predict all existing tracks
        for track in self.tracks:
            track.kf.predict()
            track.age += 1
            track.time_since_update += 1

        # 2. Association
        if len(self.tracks) > 0 and len(detections) > 0:
            track_boxes = np.array([t.bbox_3d for t in self.tracks])
            iou_matrix = iou_3d_axis_aligned(track_boxes, detections)
            cost_matrix = 1.0 - iou_matrix
            matches, unmatched_tracks, unmatched_dets = hungarian_assignment(
                cost_matrix, threshold=1.0 - self.iou_threshold
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))

        # 3. Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].kf.update(detections[det_idx])
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].time_since_update = 0
            self.tracks[track_idx].label = int(labels[det_idx])
            self.tracks[track_idx].history.append(self.tracks[track_idx].kf.position.copy())

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = self._create_track(detections[det_idx], int(labels[det_idx]))
            self.tracks.append(new_track)

        # 5. Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed or t.time_since_update == 0]

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0
