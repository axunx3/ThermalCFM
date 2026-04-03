"""Tests for the dataset module."""

import pytest
import torch
from thermal_flow.data import LogKappaTransform, ZScoreNormalize


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
