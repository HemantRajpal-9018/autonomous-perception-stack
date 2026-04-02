"""Kalman filter for 3D object state estimation.

State vector: [x, y, z, vx, vy, vz, w, l, h, yaw]
Measurement:  [x, y, z, w, l, h, yaw]
"""

import numpy as np


class KalmanFilter3D:
    """Linear Kalman filter for 3D bounding box tracking.

    Tracks position (x,y,z), velocity (vx,vy,vz), and box dimensions (w,l,h,yaw).
    Uses a constant-velocity motion model.
    """

    STATE_DIM = 10
    MEAS_DIM = 7

    def __init__(self, dt: float = 0.1, process_noise: float = 1.0,
                 measurement_noise: float = 0.5):
        self.dt = dt

        # State transition matrix F
        self.F = np.eye(self.STATE_DIM)
        # Position updates: x += vx * dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H (observe x,y,z,w,l,h,yaw from state)
        self.H = np.zeros((self.MEAS_DIM, self.STATE_DIM))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z
        self.H[3, 6] = 1  # w
        self.H[4, 7] = 1  # l
        self.H[5, 8] = 1  # h
        self.H[6, 9] = 1  # yaw

        # Process noise Q
        self.Q = np.eye(self.STATE_DIM) * process_noise
        # Lower noise for dimensions (they change slowly)
        self.Q[6, 6] = process_noise * 0.1
        self.Q[7, 7] = process_noise * 0.1
        self.Q[8, 8] = process_noise * 0.1
        self.Q[9, 9] = process_noise * 0.3

        # Measurement noise R
        self.R = np.eye(self.MEAS_DIM) * measurement_noise

        # State and covariance
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 10.0

    def initialize(self, measurement: np.ndarray) -> None:
        """Initialize state from first measurement.

        Args:
            measurement: [x, y, z, w, l, h, yaw]
        """
        self.x[0] = measurement[0]  # x
        self.x[1] = measurement[1]  # y
        self.x[2] = measurement[2]  # z
        self.x[3] = 0.0  # vx (unknown)
        self.x[4] = 0.0  # vy
        self.x[5] = 0.0  # vz
        self.x[6] = measurement[3]  # w
        self.x[7] = measurement[4]  # l
        self.x[8] = measurement[5]  # h
        self.x[9] = measurement[6]  # yaw

        # Higher initial uncertainty for velocity
        self.P = np.eye(self.STATE_DIM) * 10.0
        self.P[3, 3] = 100.0
        self.P[4, 4] = 100.0
        self.P[5, 5] = 100.0

    def predict(self) -> np.ndarray:
        """Predict next state.

        Returns:
            Predicted state vector (10,)
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement.

        Args:
            measurement: [x, y, z, w, l, h, yaw]

        Returns:
            Updated state vector (10,)
        """
        # Innovation
        y = measurement - self.H @ self.x
        # Wrap yaw angle difference to [-pi, pi]
        y[6] = (y[6] + np.pi) % (2 * np.pi) - np.pi

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        I = np.eye(self.STATE_DIM)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def dimensions(self) -> np.ndarray:
        """Returns (w, l, h)."""
        return self.x[6:9]

    @property
    def yaw(self) -> float:
        return float(self.x[9])

    @property
    def bbox_3d(self) -> np.ndarray:
        """Returns [x, y, z, w, l, h, yaw]."""
        return np.array([
            self.x[0], self.x[1], self.x[2],
            self.x[6], self.x[7], self.x[8], self.x[9],
        ])
