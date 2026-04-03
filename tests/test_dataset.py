"""Tests for the generic dataset module."""

import pytest
import torch
from thermal_flow.forward import get_forward_model
from thermal_flow.data import ThermalInverseDataset, LogKappaTransform, ZScoreNormalize


class TestThermalInverseDataset:
    def test_from_forward_model(self):
        model = get_forward_model("flash")
        ds = ThermalInverseDataset.from_forward_model(model, n_samples=32)
        assert len(ds) == 32
        sample = ds[0]
        assert "theta" in sample
        assert "y" in sample
        assert sample["theta"].shape == (model.spec.theta_dim,)
        assert sample["y"].shape == (model.spec.y_dim,)

    def test_normalization_stats(self):
        model = get_forward_model("flash")
        ds = ThermalInverseDataset.from_forward_model(model, n_samples=100)
        stats = ds.get_normalization_stats()
        assert "theta_mean" in stats
        assert "theta_std" in stats
        assert stats["theta_mean"].shape == (model.spec.theta_dim,)


class TestLogKappaTransform:
    def test_round_trip(self):
        transform = LogKappaTransform()
        kappa = torch.rand(10, 100) * 400 + 0.1
        log_kappa = transform(kappa)
        recovered = transform.inverse(log_kappa)
        assert torch.allclose(kappa, recovered, atol=1e-5)


class TestZScoreNormalize:
    def test_fit_and_transform(self):
        data = torch.randn(1000, 50)
        norm = ZScoreNormalize.fit(data)
        normalized = norm(data)
        assert normalized.mean(dim=0).abs().max() < 0.1
        assert (normalized.std(dim=0) - 1.0).abs().max() < 0.1

    def test_inverse(self):
        data = torch.randn(100, 50)
        norm = ZScoreNormalize.fit(data)
        recovered = norm.inverse(norm(data))
        assert torch.allclose(data, recovered, atol=1e-5)
