"""Extensible interpolator registry.

To add a new algorithm (e.g. kriging):
  1. Create ``core/my_method.py`` with a ``BaseInterpolator`` subclass.
  2. Decorate it with ``@register("my_method")``.
  3. Add the enum entry in ``processing/grid_algorithm.py``.
No changes to the QGIS wrapper logic are required.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

# Optional callable for reporting progress / cancellation from QGIS feedback.
# Should return True to continue, False to abort.
ProgressCallback = Callable[[float], bool] | None

_REGISTRY: dict[str, type["BaseInterpolator"]] = {}


def register(name: str):
    """Class decorator registering an interpolator under ``name``."""

    def decorator(cls: type["BaseInterpolator"]) -> type["BaseInterpolator"]:
        if name in _REGISTRY:
            raise KeyError(f"Interpolator '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_interpolator(name: str, **kwargs) -> "BaseInterpolator":
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown interpolator '{name}'. Available: {sorted(_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def available_interpolators() -> list[str]:
    return sorted(_REGISTRY)


class BaseInterpolator(ABC):
    """Common interface all gridding methods must implement."""

    @abstractmethod
    def fit(self, xy: np.ndarray, z: np.ndarray) -> None:
        """Fit the model. ``xy`` shape (N, 2), ``z`` shape (N,)."""

    @abstractmethod
    def predict(
        self, xy_out: np.ndarray, progress: ProgressCallback = None
    ) -> np.ndarray:
        """Predict values at ``xy_out`` shape (M, 2) -> (M,)."""
