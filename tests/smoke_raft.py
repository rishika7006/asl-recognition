"""Smoke test for ``src.raft_backend``.

Run from the project root:

    python3 -m tests.smoke_raft

Validates that the RAFT backend conforms to the shared interface:
    - constructable on CPU
    - flow() returns (H, W, 2) float32 at multiple input sizes
    - padding logic handles non-multiple-of-8 inputs
    - input arrays are not mutated
    - and reports CPU latency at 96x96 (sanity check, not pass/fail).
"""

from __future__ import annotations

import time

import numpy as np

from src import raft_backend


def _make_random_pair(size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create two random uint8 RGB frames of shape (size, size, 3)."""
    rng = np.random.default_rng(seed)
    prev = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    nxt = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return prev, nxt


def _check_shape_and_dtype(out: np.ndarray, h: int, w: int, label: str) -> None:
    assert out.shape == (h, w, 2), (
        f"[{label}] expected shape ({h}, {w}, 2), got {out.shape}"
    )
    assert out.dtype == np.float32, (
        f"[{label}] expected dtype float32, got {out.dtype}"
    )


def _check_inputs_unmodified(
    prev_orig: np.ndarray,
    next_orig: np.ndarray,
    prev_after: np.ndarray,
    next_after: np.ndarray,
    label: str,
) -> None:
    assert np.array_equal(prev_orig, prev_after), f"[{label}] prev frame was mutated"
    assert np.array_equal(next_orig, next_after), f"[{label}] next frame was mutated"


def main() -> int:
    print(f"raft_backend.NAME = {raft_backend.NAME!r}")
    assert raft_backend.NAME == "raft", "NAME must be 'raft'"

    # 1. Construct estimator on CPU.
    print("constructing estimator on CPU...")
    estimator = raft_backend.make_estimator(device="cpu")
    assert estimator is not None, "make_estimator returned None"
    assert "model" in estimator and "device" in estimator, (
        "estimator missing expected keys"
    )
    print(f"  device={estimator['device']}, model={type(estimator['model']).__name__}")

    # 2. flow() at 96x96 (multiple of 8).
    prev, nxt = _make_random_pair(96, seed=1)
    prev_copy, nxt_copy = prev.copy(), nxt.copy()
    out = raft_backend.flow(estimator, prev, nxt)
    _check_shape_and_dtype(out, 96, 96, "96x96")
    _check_inputs_unmodified(prev_copy, nxt_copy, prev, nxt, "96x96")
    print(f"  96x96: out shape={out.shape}, dtype={out.dtype}, "
          f"|u| mean={np.mean(np.abs(out[..., 0])):.3f}, "
          f"|v| mean={np.mean(np.abs(out[..., 1])):.3f}")

    # 3. flow() at 100x100 (NOT a multiple of 8) — exercises padding/crop.
    prev, nxt = _make_random_pair(100, seed=2)
    prev_copy, nxt_copy = prev.copy(), nxt.copy()
    out = raft_backend.flow(estimator, prev, nxt)
    _check_shape_and_dtype(out, 100, 100, "100x100")
    _check_inputs_unmodified(prev_copy, nxt_copy, prev, nxt, "100x100")
    print(f"  100x100: out shape={out.shape}, dtype={out.dtype} (padding path OK)")

    # 4. flow() at 64x64 (smaller, multiple of 8).
    prev, nxt = _make_random_pair(64, seed=3)
    prev_copy, nxt_copy = prev.copy(), nxt.copy()
    out = raft_backend.flow(estimator, prev, nxt)
    _check_shape_and_dtype(out, 64, 64, "64x64")
    _check_inputs_unmodified(prev_copy, nxt_copy, prev, nxt, "64x64")
    print(f"  64x64: out shape={out.shape}, dtype={out.dtype}")

    # 5. CPU latency benchmark at 96x96 over 10 runs.
    prev, nxt = _make_random_pair(96, seed=42)
    # Warmup (first call has lazy-init overhead).
    _ = raft_backend.flow(estimator, prev, nxt)
    timings_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = raft_backend.flow(estimator, prev, nxt)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms_arr = np.array(timings_ms)
    mean_ms = float(timings_ms_arr.mean())
    p95_ms = float(np.percentile(timings_ms_arr, 95))
    print(
        f"CPU latency @ 96x96, n=10, num_flow_updates=6: "
        f"mean={mean_ms:.1f} ms, p95={p95_ms:.1f} ms "
        f"(individual: {[f'{t:.0f}' for t in timings_ms]})"
    )

    print("OK: smoke_raft passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
