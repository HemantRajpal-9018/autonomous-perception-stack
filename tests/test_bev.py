"""Tests for BEV transform modules."""

import torch
import pytest


class TestHomographyBEV:
    def test_init(self):
        from perception_stack.bev.homography import HomographyBEV

        model = HomographyBEV(
            image_size=(224, 400), bev_size=(100, 100),
            feature_channels=32, num_cameras=4,
        )
        assert model.num_cameras == 4
        assert model.bev_size == (100, 100)

    def test_forward_shape(self):
        from perception_stack.bev.homography import HomographyBEV

        B, N, C, H, W = 2, 4, 3, 224, 400
        model = HomographyBEV(
            image_size=(H, W), bev_size=(100, 100),
            feature_channels=32, num_cameras=N,
        )
        images = torch.randn(B, N, C, H, W)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        intrinsics[:, :, 0, 0] = 500
        intrinsics[:, :, 1, 1] = 500
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        out = model(images, intrinsics, extrinsics)
        assert out.shape == (B, 32, 100, 100)

    def test_compute_homography(self):
        from perception_stack.bev.homography import HomographyBEV

        model = HomographyBEV()
        intrinsic = torch.eye(3).unsqueeze(0)
        intrinsic[0, 0, 0] = 500
        intrinsic[0, 1, 1] = 500
        extrinsic = torch.eye(4).unsqueeze(0)

        H = model.compute_homography(intrinsic, extrinsic)
        assert H.shape == (1, 3, 3)

    def test_warp_to_bev(self):
        from perception_stack.bev.homography import HomographyBEV

        model = HomographyBEV(bev_size=(50, 50), feature_channels=32)
        features = torch.randn(1, 32, 28, 50)
        homography = torch.eye(3).unsqueeze(0)

        bev = model.warp_to_bev(features, homography)
        assert bev.shape == (1, 32, 50, 50)


class TestLiftSplatShoot:
    def test_init(self):
        from perception_stack.bev.lift_splat import LiftSplatShoot

        model = LiftSplatShoot(
            image_size=(224, 400), bev_size=(100, 100),
            feature_channels=32, num_depth_bins=20,
        )
        assert model.num_depth_bins == 20
        assert model.feature_channels == 32

    def test_create_frustum(self):
        from perception_stack.bev.lift_splat import LiftSplatShoot

        model = LiftSplatShoot(num_depth_bins=10)
        frustum = model.create_frustum(14, 25, torch.device("cpu"))
        assert frustum.shape == (10, 14, 25, 3)

    def test_depth_net(self):
        from perception_stack.bev.lift_splat import DepthNet

        net = DepthNet(64, num_depth_bins=20)
        x = torch.randn(2, 64, 14, 25)
        out = net(x)
        assert out.shape == (2, 20, 14, 25)
