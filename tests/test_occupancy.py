"""Tests for occupancy grid module."""

import numpy as np
import pytest


class TestOccupancyGrid:
    def test_init(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid(width=100, height=100, resolution=0.5)
        assert grid.width == 100
        assert grid.height == 100
        assert grid.log_odds.shape == (100, 100)

    def test_world_to_grid(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid(origin=(-50.0, -50.0), resolution=0.5)
        gx, gy = grid.world_to_grid(0.0, 0.0)
        assert gx == 100
        assert gy == 100

    def test_grid_to_world(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid(origin=(-50.0, -50.0), resolution=0.5)
        x, y = grid.grid_to_world(100, 100)
        assert abs(x - 0.0) < 1e-5
        assert abs(y - 0.0) < 1e-5

    def test_probability_grid(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid()
        prob = grid.probability_grid
        # All cells should be 0.5 (unknown) initially
        assert abs(prob.mean() - 0.5) < 1e-5

    def test_update_from_detections(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid()
        boxes = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
        grid.update_from_detections(boxes)

        # Cells at (0,0) should be more occupied
        prob = grid.probability_grid
        gx, gy = grid.world_to_grid(0.0, 0.0)
        assert prob[gy, gx] > 0.5

    def test_update_from_lidar(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid(width=50, height=50, origin=(-12.5, -12.5))
        # Simple point cloud: a few points ahead
        points = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, -1.0, 0.0]])
        sensor_pos = np.array([0.0, 0.0])
        grid.update_from_lidar(points, sensor_pos)

        prob = grid.probability_grid
        # Point location should be more occupied
        gx, gy = grid.world_to_grid(5.0, 0.0)
        if grid.is_valid(gx, gy):
            assert prob[gy, gx] > 0.5

    def test_decay(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid(decay_rate=0.5)
        grid.log_odds[50, 50] = 3.0
        grid.decay()
        assert abs(grid.log_odds[50, 50] - 1.5) < 1e-5

    def test_costmap(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid()
        costmap = grid.get_costmap()
        assert costmap.shape == (grid.height, grid.width)
        assert costmap.min() >= 0.0
        assert costmap.max() <= 1.0

    def test_reset(self):
        from perception_stack.occupancy.grid import OccupancyGrid

        grid = OccupancyGrid()
        grid.log_odds[10, 10] = 5.0
        grid.reset()
        assert grid.log_odds.sum() == 0.0
