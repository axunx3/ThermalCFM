"""Tests for Flow Matching components with generic dimensions."""

import pytest
import torch
from thermal_flow.forward import get_forward_model
from thermal_flow.models import VelocityNet, ConditionalFlowMatching
from thermal_flow.models.velocity_net import SinusoidalTimeEmbedding


class TestVelocityNet:
    def test_output_shape_explicit(self):
        net = VelocityNet(x_dim=100, y_dim=120)
        x = torch.randn(4, 100)
        t = torch.rand(4)
        y = torch.randn(4, 120)
        v = net(x, t, y)
        assert v.shape == (4, 100)

    def test_from_spec(self):
        """VelocityNet should auto-configure from ForwardModelSpec."""
        model = get_forward_model("flash")
        net = VelocityNet.from_spec(model.spec)
        # Flash: theta_dim=2, y_dim=200
        x = torch.randn(4, 2)
        t = torch.rand(4)
        y = torch.randn(4, 200)
        v = net(x, t, y)
        assert v.shape == (4, 2)

    def test_auto_scaling(self):
        """Small problems should get smaller networks."""
        model = get_forward_model("flash")
        net = VelocityNet.from_spec(model.spec)
        n_params = sum(p.numel() for p in net.parameters())
        # Flash (2D) should be much smaller than a default 512x4 network
        assert n_params < 1_000_000

    def test_time_embedding(self):
        embed = SinusoidalTimeEmbedding(128)
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = embed(t)
        assert emb.shape == (3, 128)


class TestConditionalFlowMatching:
    def test_loss_computation(self):
        net = VelocityNet(x_dim=50, y_dim=60)
        cfm = ConditionalFlowMatching(net)
        x_1 = torch.randn(8, 50)
        y = torch.randn(8, 60)
        loss = cfm.compute_loss(x_1, y)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_with_flash_dims(self):
        """CFM should work with Flash model dimensions."""
        model = get_forward_model("flash")
        net = VelocityNet.from_spec(model.spec)
        cfm = ConditionalFlowMatching(net)

        data = model.generate_dataset(16)
        theta = model.log_transform(data["theta"]).float()
        y = data["y_noisy"].float()

        loss = cfm.compute_loss(theta, y)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_sample_shape(self):
        net = VelocityNet(x_dim=2, y_dim=200)
        cfm = ConditionalFlowMatching(net)
        y = torch.randn(4, 200)
        samples = cfm.sample(y, n_steps=5, n_samples=1)
        assert samples.shape == (4, 2)

    def test_sample_multiple(self):
        net = VelocityNet(x_dim=2, y_dim=200)
        cfm = ConditionalFlowMatching(net)
        y = torch.randn(4, 200)
        samples = cfm.sample(y, n_steps=5, n_samples=8)
        assert samples.shape == (32, 2)  # 4 * 8
