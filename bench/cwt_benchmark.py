"""Barebones speed + sanity-check comparison: torch_cwt vs ptwt vs
ssqueezepy (GPU backend), on synthetic dummy data only.

Usage: python cwt_benchmark.py [--devices cpu,cuda] [--channels 23]
                                [--fs 256] [--durations 10,60,300]

Prints one fixed-width stdout table per device: backend, shape, total_s,
ms/channel, peak_MiB (CUDA only). No CSV/results-pipeline output.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from make_dummy_data import make_dummy_signals

import torchcwt

F0, F1, FN = 8.0, 40.0, 8  # matches EEG_Benchmarks' current CWT band assumption


def available_devices(requested: list[str]) -> list[str]:
    out = []
    for d in requested:
        if d == "cpu":
            out.append(d)
        elif d == "cuda" and torch.cuda.is_available():
            out.append(d)
        elif d == "mps" and torch.backends.mps.is_available():
            out.append(d)
    return out


def _reset_mem(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mib(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2)
    return None


# --- backend adapters -------------------------------------------------

def run_torch_cwt(signals: np.ndarray, fs: float, device: str) -> np.ndarray:
    coeffs, _ = torchcwt.transform_batch(signals, fs, F1, F0, FN, device=device)
    return coeffs  # [n_channels, FN, T] complex64


def run_ptwt(signals: np.ndarray, fs: float, device: str) -> np.ndarray:
    import pywt
    import ptwt

    wavelet = "cmor1.5-1.0"  # complex Morlet, closest ptwt/pywt analogue
    central_freq = pywt.central_frequency(wavelet)
    freqs = np.geomspace(F0, F1, FN)[::-1]
    scales = central_freq * fs / freqs

    x = torch.from_numpy(np.ascontiguousarray(signals, dtype=np.float32)).to(device)
    coeffs, _ = ptwt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    # ptwt returns [n_scales, n_channels, T]; reorder to [n_channels, n_scales, T]
    return coeffs.permute(1, 0, 2).detach().cpu().numpy()


def run_ssqueezepy(signals: np.ndarray, fs: float, device: str) -> np.ndarray:
    import os

    os.environ["SSQ_GPU"] = "1" if device == "cuda" else "0"
    import ssqueezepy as ssq

    # Same scale=fs/freq convention torch_cwt documents (cwt.py's own
    # "scale = sampling_rate/freq"), passed explicitly so ssqueezepy uses
    # our FN=8 grid instead of its own auto "scales='log'" (360 scales) --
    # not a bit-exact wavelet match (different center-frequency
    # normalization per library, same caveat as the ptwt adapter), but same
    # frequency axis for a fair shape/throughput comparison.
    freqs = np.geomspace(F0, F1, FN)[::-1]
    scales = fs / freqs

    out = []
    for ch in range(signals.shape[0]):
        cwt_matrix, *_ = ssq.cwt(signals[ch], wavelet="morlet", fs=fs, scales=scales)
        # SSQ_GPU=1 returns a live CUDA torch tensor, not a numpy array --
        # move it home before stacking (this is what actually crashed the
        # cuda path: np.stack() on a CUDA tensor raises, it never OOMed).
        if hasattr(cwt_matrix, "detach"):
            cwt_matrix = cwt_matrix.detach().cpu().numpy()
        out.append(cwt_matrix)
    return np.stack(out, axis=0)


BACKENDS = {
    "torch_cwt": run_torch_cwt,
    "ptwt": run_ptwt,
    "ssqueezepy": run_ssqueezepy,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="cuda")  # GPU is the comparison that matters at hour-scale durations; pass --devices cpu,cuda for a small-scale CPU sanity check
    ap.add_argument("--channels", type=int, default=23)
    ap.add_argument("--fs", type=float, default=256.0)
    ap.add_argument("--durations", default="60,300,600")  # 1min/5min/10min -- enough to see which backend is fastest
    ap.add_argument("--backends", default=",".join(BACKENDS))
    args = ap.parse_args()

    devices = available_devices(args.devices.split(","))
    durations = [float(d) for d in args.durations.split(",")]
    backends = [b for b in args.backends.split(",") if b in BACKENDS]

    print(f"{'backend':<12} {'device':<6} {'duration_s':>10} {'shape':>18} "
          f"{'total_s':>9} {'ms/ch':>8} {'peak_MiB':>9}", flush=True)

    for duration_s in durations:
        signals = make_dummy_signals(args.channels, args.fs, duration_s)
        for device in devices:
            for name in backends:
                fn = BACKENDS[name]
                try:
                    print(f"... starting {name} on {device} for duration_s={duration_s:.0f}", flush=True)
                    _reset_mem(device)
                    t0 = time.perf_counter()
                    out = fn(signals, args.fs, device)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    total_s = time.perf_counter() - t0
                    peak = _peak_mib(device)
                    peak_str = f"{peak:9.1f}" if peak is not None else "      n/a"
                    print(f"{name:<12} {device:<6} {duration_s:>10.1f} {str(out.shape):>18} "
                          f"{total_s:>9.3f} {total_s / args.channels * 1000:>8.2f} {peak_str}", flush=True)
                except Exception as exc:  # keep going -- a broken competitor shouldn't kill the sweep
                    print(f"{name:<12} {device:<6} {duration_s:>10.1f} {'FAILED':>18} "
                          f"{'':>9} {'':>8} {'':>9}  ({exc})", flush=True)


if __name__ == "__main__":
    main()
