"""Tests for lane detection modules."""

import numpy as np
import torch
import pytest


class TestLaneSegmentationNet:
    def test_forward_shape(self):
        from perception_stack.lanes.segmentation import LaneSegmentationNet

        model = LaneSegmentationNet(num_classes=3, embedding_dim=4)
        x = torch.randn(2, 3, 256, 512)
        out = model(x)

        assert "segmentation" in out
        assert "embedding" in out
        assert out["segmentation"].shape == (2, 3, 256, 512)
        assert out["embedding"].shape == (2, 4, 256, 512)

    def test_different_input_sizes(self):
        from perception_stack.lanes.segmentation import LaneSegmentationNet

        model = LaneSegmentationNet(num_classes=3)
        # Should handle different input sizes
        for h, w in [(128, 256), (64, 128)]:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            assert out["segmentation"].shape == (1, 3, h, w)


class TestPolynomialLaneFitter:
    def test_fit_polynomial(self):
        from perception_stack.lanes.polynomial import PolynomialLaneFitter

        fitter = PolynomialLaneFitter(degree=3)
        # Create a known polynomial: x = 0.001*y^3 - 0.1*y^2 + y + 100
        y = np.linspace(100, 700, 200)
        x = 0.001 * y ** 3 - 0.1 * y ** 2 + y + 100 + np.random.randn(200) * 2
        points = np.stack([y, x], axis=1)

        coeffs = fitter.fit_polynomial(points)
        assert coeffs is not None
        assert len(coeffs) == 4  # degree + 1

    def test_evaluate_polynomial(self):
        from perception_stack.lanes.polynomial import PolynomialLaneFitter

        fitter = PolynomialLaneFitter(degree=2, image_height=720)
        coeffs = np.array([0.001, -0.5, 300])

        points = fitter.evaluate_polynomial(coeffs)
        assert points.shape == (720, 2)
        assert points[0, 0] == 0.0  # y starts at 0

    def test_fit_lanes_from_mask(self):
        from perception_stack.lanes.polynomial import PolynomialLaneFitter

        fitter = PolynomialLaneFitter(degree=2, num_lanes=2, min_points=20)

        # Create synthetic lane mask
        mask = np.zeros((200, 400), dtype=np.int32)
        for y in range(50, 200):
            x1 = int(150 + 0.5 * (y - 100))
            x2 = int(250 + 0.5 * (y - 100))
            if 0 <= x1 < 400:
                mask[y, max(0, x1 - 2):min(400, x1 + 3)] = 1
            if 0 <= x2 < 400:
                mask[y, max(0, x2 - 2):min(400, x2 + 3)] = 1

        results = fitter.fit_lanes(mask)
        assert len(results) >= 1  # At least one lane detected
        for lane in results:
            assert "coefficients" in lane
            assert "confidence" in lane
            assert lane["confidence"] >= 0

    def test_extract_no_lanes(self):
        from perception_stack.lanes.polynomial import PolynomialLaneFitter

        fitter = PolynomialLaneFitter()
        mask = np.zeros((200, 400), dtype=np.int32)
        lanes = fitter.extract_lane_pixels(mask)
        assert len(lanes) == 0
