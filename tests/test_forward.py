"""Tests for the Borca-Tasciuc forward model."""

import pytest
import torch
from thermal_flow.forward import BorcaTasciucModel, ProfileGenerator, NoiseModel


class TestBorcaTasciucModel:
    """Tests for the 3-omega forward model."""

    def test_model_initialization(self):
        model = BorcaTasciucModel()
        assert model.n_freq == 60
        assert model.freqs.shape == (60,)
        assert model.k_nodes.shape == (64,)

    @pytest.mark.skip(reason="Forward model not yet implemented")
    def test_homogeneous_substrate(self):
        """For a homogeneous substrate, the result should match
        the analytical Cahill formula."""
        model = BorcaTasciucModel()
        # Homogeneous Si: κ = 148 W/mK
        kappa = torch.full((1, 100), 148.0)
        rho_cp = torch.full((1, 100), 1.63e6)  # Si volumetric heat capacity
        thickness = torch.full((1, 100), 0.5e-6)

        result = model(kappa, rho_cp, thickness)
        assert result.shape == (1, 60)

    @pytest.mark.skip(reason="Forward model not yet implemented")
    def test_differentiability(self):
        """Forward model should be differentiable w.r.t. kappa."""
        model = BorcaTasciucModel()
        kappa = torch.randn(2, 100, requires_grad=True)
        rho_cp = torch.full((2, 100), 1.63e6)
        thickness = torch.full((2, 100), 0.5e-6)

        result = model(kappa, rho_cp, thickness)
        loss = result.abs().sum()
        loss.backward()
        assert kappa.grad is not None


class TestProfileGenerator:
    """Tests for κ(z) profile generation."""

    def test_initialization(self):
        gen = ProfileGenerator(n_layers=50, z_max=20e-6)
        assert gen.n_layers == 50
        assert len(gen.z_grid) == 50


class TestNoiseModel:
    """Tests for measurement noise."""

    def test_initialization(self):
        noise = NoiseModel(real_fraction=0.03, imag_fraction=0.07)
        assert noise.real_fraction == 0.03
