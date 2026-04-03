from .base import ForwardModel, ForwardModelSpec
from .flash import FlashModel
from .borca_tasciuc import ThreeOmegaModel
from .tdtr import TDTRModel
from .profiles import ProfileGenerator
from .noise import NoiseModel

# Registry of all available forward models
FORWARD_MODELS = {
    "flash": FlashModel,
    "3omega": ThreeOmegaModel,
    "tdtr": TDTRModel,
}


def get_forward_model(name: str, **kwargs) -> ForwardModel:
    """Get a forward model by name.

    Args:
        name: One of 'flash', '3omega', 'tdtr'.
        **kwargs: Model-specific arguments.

    Returns:
        Instantiated ForwardModel.
    """
    if name not in FORWARD_MODELS:
        raise ValueError(
            f"Unknown forward model '{name}'. Available: {list(FORWARD_MODELS.keys())}"
        )
    return FORWARD_MODELS[name](**kwargs)


__all__ = [
    "ForwardModel",
    "ForwardModelSpec",
    "FlashModel",
    "ThreeOmegaModel",
    "TDTRModel",
    "ProfileGenerator",
    "NoiseModel",
    "FORWARD_MODELS",
    "get_forward_model",
]
