"""Tests for visualization modules."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")


class TestBoxes3D:
    def test_corners_from_bbox3d(self):
        from perception_stack.visualization.boxes_3d import corners_from_bbox3d

        box = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0])
        corners = corners_from_bbox3d(box)
        assert corners.shape == (8, 3)

    def test_draw_3d_boxes(self):
        from perception_stack.visualization.boxes_3d import draw_3d_boxes

        boxes = np.array([
            [10.0, 5.0, 0.0, 2.0, 4.0, 1.5, 0.1],
            [-5.0, 3.0, 0.0, 0.6, 0.6, 1.8, 0.0],
        ])
        labels = np.array([0, 1])
        img = draw_3d_boxes(boxes, labels, image_size=(480, 640))
        assert img.shape == (480, 640, 3)


class TestBEVPlot:
    def test_plot_bev_basic(self, sample_boxes, sample_labels):
        from perception_stack.visualization.bev_plot import plot_bev
        import matplotlib.pyplot as plt

        fig = plot_bev(boxes=sample_boxes, labels=sample_labels)
        assert fig is not None
        plt.close(fig)

    def test_plot_bev_with_point_cloud(self, sample_point_cloud):
        from perception_stack.visualization.bev_plot import plot_bev
        import matplotlib.pyplot as plt

        fig = plot_bev(point_cloud=sample_point_cloud)
        assert fig is not None
        plt.close(fig)


class TestPointCloudViz:
    def test_render_bev(self, sample_point_cloud):
        from perception_stack.visualization.point_cloud_viz import render_point_cloud_bev
        import matplotlib.pyplot as plt

        fig = render_point_cloud_bev(sample_point_cloud)
        assert fig is not None
        plt.close(fig)

    def test_render_bev_by_intensity(self, sample_point_cloud):
        from perception_stack.visualization.point_cloud_viz import render_point_cloud_bev
        import matplotlib.pyplot as plt

        fig = render_point_cloud_bev(sample_point_cloud, color_by="intensity")
        assert fig is not None
        plt.close(fig)
