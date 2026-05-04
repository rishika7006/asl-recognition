"""Dispatcher that registers all available optical-flow backends.

Each backend module exposes the same interface:

    NAME: str
    make_estimator(device: str = "cpu") -> Any | None
    flow(estimator, prev_rgb, next_rgb) -> np.ndarray  # (H, W, 2) float32

Use ``get_backend(name)`` to look up a backend by string name from a
config file or CLI flag. The RAFT backend is loaded lazily and is
optional — if torchvision isn't installed or ``raft_backend`` hasn't
been written yet, only Farnebäck will be registered.
"""

from __future__ import annotations

from typing import Any, Dict

from . import farneback_backend

# Map of backend name -> backend module (must expose NAME, make_estimator, flow).
BACKENDS: Dict[str, Any] = {farneback_backend.NAME: farneback_backend}

# RAFT is optional. The import may fail if torchvision is missing, if the
# weights can't be downloaded, or simply if Agent B's raft_backend.py
# doesn't exist yet during parallel development. In any of those cases
# we silently fall back to Farnebäck-only.
try:  # pragma: no cover - import-time branch
    from . import raft_backend  # type: ignore[attr-defined]

    BACKENDS[raft_backend.NAME] = raft_backend
except Exception:  # noqa: BLE001 - intentionally broad: any import failure is OK
    pass


def get_backend(name: str) -> Any:
    """Look up a backend module by name.

    Args:
        name: Backend identifier, e.g. ``"farneback"`` or ``"raft"``.

    Returns:
        The backend module (with NAME, make_estimator, flow attributes).

    Raises:
        ValueError: If ``name`` is not a registered backend.
    """
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(BACKENDS.keys())}"
        )
    return BACKENDS[name]


def available_backends() -> list[str]:
    """Return the list of registered backend names (sorted)."""
    return sorted(BACKENDS.keys())
