"""Tests for Flow Matching components."""

import pytest
import torch
from thermal_flow.models import VelocityNet, ConditionalFlowMatching


class TestVelocityNet:
    def test_output_shape(self):
        net = VelocityNet(x_dim=100, y_dim=120)
        x = torch.randn(4, 100)
        t = torch.rand(4)
        y = torch.randn(4, 120)
        v = net(x, t, y)
        assert v.shape == (4, 100)

    def test_time_embedding(self):
        from thermal_flow.models.velocity_net import SinusoidalTimeEmbedding
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
