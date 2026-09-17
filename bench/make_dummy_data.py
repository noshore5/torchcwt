"""Synthetic dummy-data generator for the CWT backend benchmark.

No real EEG data dependency -- pure numpy sine+noise mixes, parametrized by
n_channels / sampling_rate / duration_s so the benchmark can sweep realistic
EEG-scale shapes (e.g. 23 channels x 256 Hz x 1-60 min) without pulling in
any dataset loading code.
"""

from __future__ import annotations

import numpy as np


def make_dummy_signals(
    n_channels: int,
    sampling_rate: float,
    duration_s: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Returns float32 array [n_channels, n_time] -- a sine/chirp mix plus
    noise, distinct (but reproducible) per channel via `seed`.
    """
    rng = np.random.default_rng(seed)
    n_time = int(round(duration_s * sampling_rate))
    t = np.arange(n_time, dtype=np.float64) / sampling_rate

    signals = np.empty((n_channels, n_time), dtype=np.float32)
    for ch in range(n_channels):
        f_lo = rng.uniform(8.0, 14.0)
        f_hi = rng.uniform(20.0, 40.0)
        chirp = np.sin(2 * np.pi * (f_lo * t + (f_hi - f_lo) * t**2 / (2 * duration_s)))
        tone = 0.5 * np.sin(2 * np.pi * rng.uniform(8.0, 40.0) * t + rng.uniform(0, 2 * np.pi))
        noise = 0.1 * rng.standard_normal(n_time)
        signals[ch] = (chirp + tone + noise).astype(np.float32)

    return signals


if __name__ == "__main__":
    x = make_dummy_signals(n_channels=23, sampling_rate=256.0, duration_s=10.0)
    print(f"dummy signals shape={x.shape} dtype={x.dtype}")
