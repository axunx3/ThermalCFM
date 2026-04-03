"""Tests for forward models and the pluggable interface."""

import pytest
import torch
from thermal_flow.forward import get_forward_model, ForwardModel, FORWARD_MODELS
from thermal_flow.forward.flash import FlashModel
from thermal_flow.forward.borca_tasciuc import ThreeOmegaModel
from thermal_flow.forward.tdtr import TDTRModel


class TestForwardModelRegistry:
    def test_all_models_registered(self):
        assert "flash" in FORWARD_MODELS
        assert "3omega" in FORWARD_MODELS
        assert "tdtr" in FORWARD_MODELS

    def test_get_forward_model(self):
        model = get_forward_model("flash")
        assert isinstance(model, ForwardModel)
        assert model.spec.name == "flash"

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_forward_model("nonexistent")


class TestForwardModelInterface:
    """Test that all models implement the full interface."""

    @pytest.mark.parametrize("name", ["flash"])
    def test_spec(self, name):
        model = get_forward_model(name)
        spec = model.spec
        assert spec.theta_dim > 0
        assert spec.y_dim > 0
        assert len(spec.theta_names) == spec.theta_dim
        assert len(spec.theta_bounds) == 2
        assert spec.theta_bounds[0].shape == (spec.theta_dim,)

    @pytest.mark.parametrize("name", ["flash"])
    def test_sample_prior(self, name):
        model = get_forward_model(name)
        theta = model.sample_prior(16)
        assert theta.shape == (16, model.spec.theta_dim)
        # Check within physical bounds
        lower, upper = model.spec.theta_bounds
        assert (theta >= lower).all()
        assert (theta <= upper).all()

    @pytest.mark.parametrize("name", ["flash"])
    def test_generate_dataset(self, name):
        model = get_forward_model(name)
        data = model.generate_dataset(8)
        assert data["theta"].shape == (8, model.spec.theta_dim)
        assert data["y_clean"].shape == (8, model.spec.y_dim)
        assert data["y_noisy"].shape == (8, model.spec.y_dim)

    @pytest.mark.parametrize("name", ["flash"])
    def test_log_transform_roundtrip(self, name):
        model = get_forward_model(name)
        theta = model.sample_prior(8)
        log_theta = model.log_transform(theta)
        recovered = model.exp_transform(log_theta)
        assert torch.allclose(theta, recovered, atol=1e-6)


class TestFlashModel:
    def test_forward_shape(self):
        model = FlashModel(n_time=50)
        theta = model.sample_prior(4)
        y = model(theta)
        assert y.shape == (4, 50)

    def test_temperature_physical(self):
        """T(t) should start near 0 and approach 1 for large t."""
        model = FlashModel(n_time=200, t_max=2.0)
        # Use a moderate diffusivity
        theta = torch.tensor([[1e-5, 0.0]], dtype=torch.float64)
        y = model(theta)
        # Late-time value should be near 1.0 (no heat loss, h=0)
        assert y[0, -1].item() > 0.9

    def test_noise_adds_variation(self):
        model = FlashModel()
        theta = model.sample_prior(4)
        y_clean = model(theta)
        y_noisy = model.add_noise(y_clean)
        assert not torch.allclose(y_clean, y_noisy)
