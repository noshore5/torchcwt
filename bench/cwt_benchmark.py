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
    # scales defaults to float64 (numpy); ptwt promotes its filter bank and
    # output to match, giving complex128 output even though x is float32 --
    # that's not a fair comparison against torch_cwt/ssqueezepy's complex64
    # (costs ptwt extra memory+time it wouldn't need at matched precision).
    # Force float32 scales so ptwt builds float32 filters -> complex64 out.
    scales_t = torch.as_tensor(scales, dtype=torch.float32, device=device)
    coeffs, _ = ptwt.cwt(x, scales_t, wavelet, sampling_period=1.0 / fs)
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

    # ssqueezepy.cwt() natively batches a 2-D [n_channels, T] input (its
    # default vectorized=True path) -- looping per-channel in Python here
    # was paying ssqueezepy's per-call setup cost (wavelet/filter-bank
    # construction, backend dispatch) 23x over, which dominated the timing
    # at short durations. Passing the whole array in one call amortizes
    # that setup once, like torch_cwt's and ptwt's batched adapters do.
    cwt_matrix, *_ = ssq.cwt(signals, wavelet="morlet", fs=fs, scales=scales)
    # SSQ_GPU=1 returns a live CUDA torch tensor, not a numpy array -- move
    # it home before returning (this is what actually crashed the cuda
    # path before: np.stack() on a CUDA tensor raises, it never OOMed).
    if hasattr(cwt_matrix, "detach"):
        cwt_matrix = cwt_matrix.detach().cpu().numpy()
    return cwt_matrix


BACKENDS = {
    "torch_cwt": run_torch_cwt,
    "ptwt": run_ptwt,
    "ssqueezepy": run_ssqueezepy,
}


def _magnitude_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of |CWT| between two backends' outputs, flattened.
    Not a bit-exact parity check (wavelet normalization differs per library,
    see adapter comments) -- just enough to confirm nobody is producing
    garbage on the same frequency grid.
    """
    ma, mb = np.abs(a).ravel(), np.abs(b).ravel()
    if ma.shape != mb.shape:
        return float("nan")
    ma = ma - ma.mean()
    mb = mb - mb.mean()
    denom = np.linalg.norm(ma) * np.linalg.norm(mb)
    return float((ma @ mb) / denom) if denom > 0 else float("nan")


def _warm_up(backends: list[str], device: str, fs: float, channels: int) -> None:
    """Run every backend once on a tiny throwaway signal before any timed
    call, on every device that will be used. Without this, whichever
    backend happens to run first in the sweep eats a one-time CUDA
    context / cuFFT-plan-cache / JIT warm-up cost that later backends
    don't pay -- that's what made torch_cwt look artificially slow at the
    smallest duration in earlier runs. Warming every backend up-front
    makes the timed sweep itself apples-to-apples.
    """
    tiny = make_dummy_signals(channels, fs, 4.0, seed=999)
    for name in backends:
        try:
            BACKENDS[name](tiny, fs, device)
            if device == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"... warm-up failed for {name} on {device}: {exc}", flush=True)


def main() -> None:
    global FN
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="cuda")  # GPU is the comparison that matters at hour-scale durations; pass --devices cpu,cuda for a small-scale CPU sanity check
    ap.add_argument("--channels", type=int, default=23)
    ap.add_argument("--fs", type=float, default=256.0)
    ap.add_argument("--durations", default="60,300,600")  # 1min/5min/10min -- enough to see which backend is fastest
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--n-scales", type=int, default=FN,
                     help="frequency scales in the CWT grid (default 8, matching "
                          "EEG_Benchmarks' current band assumption); memory scales "
                          "roughly linearly with n_scales x duration_s")
    args = ap.parse_args()
    FN = args.n_scales

    devices = available_devices(args.devices.split(","))
    durations = [float(d) for d in args.durations.split(",")]
    backends = [b for b in args.backends.split(",") if b in BACKENDS]

    print(f"{'backend':<12} {'device':<6} {'duration_s':>10} {'shape':>18} {'dtype':>10} "
          f"{'total_s':>9} {'ms/ch':>8} {'peak_MiB':>9}", flush=True)

    # Correctness sanity check on a small shared signal, once per device,
    # before any timing -- pairwise magnitude correlation across backends
    # on the same frequency grid. This isn't wavelet-normalization-exact
    # (see adapter comments) but catches anyone producing garbage.
    for device in devices:
        _warm_up(backends, device, args.fs, args.channels)
        probe = make_dummy_signals(args.channels, args.fs, 4.0, seed=1)
        outs = {}
        for name in backends:
            try:
                outs[name] = BACKENDS[name](probe, args.fs, device)
            except Exception as exc:
                print(f"... correctness probe failed for {name} on {device}: {exc}", flush=True)
        names = list(outs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                corr = _magnitude_corr(outs[a], outs[b])
                # Also test with b's frequency/scale axis (axis=1, shape is
                # [n_channels, n_scales, T] for every adapter here) reversed
                # -- if some backend silently re-sorts scales internally
                # despite being handed the same freqs/scales array as the
                # others, this flip will jump close to +1 while the
                # unflipped correlation stays negative/near-zero, nailing
                # down "scale-axis order mismatch" vs. "actually different
                # computation" without guessing.
                corr_flipped = _magnitude_corr(outs[a], outs[b][:, ::-1, :])
                flag = "  <-- LOOKS LIKE A SCALE-AXIS ORDER FLIP" if corr_flipped > corr + 0.3 else ""
                print(f"... |CWT| correlation {a} vs {b} on {device}: {corr:.3f} "
                      f"(axis-flipped: {corr_flipped:.3f}){flag}", flush=True)

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
                    print(f"{name:<12} {device:<6} {duration_s:>10.1f} {str(out.shape):>18} {str(out.dtype):>10} "
                          f"{total_s:>9.3f} {total_s / args.channels * 1000:>8.2f} {peak_str}", flush=True)
                except Exception as exc:  # keep going -- a broken competitor shouldn't kill the sweep
                    print(f"{name:<12} {device:<6} {duration_s:>10.1f} {'FAILED':>18} {'':>10} "
                          f"{'':>9} {'':>8} {'':>9}  ({exc})", flush=True)


if __name__ == "__main__":
    main()
