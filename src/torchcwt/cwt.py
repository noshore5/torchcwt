"""Torch-native continuous wavelet transform (Morlet), computed via the
Fourier convolution theorem in torch.fft -- a from-scratch, torch-only
numerical replacement for coherence_utils.py's FFTW-backed `fcwt.cwt()`
call. No dependency on the `fcwt` package or its FFTW backend.

Algorithm: the CWT of a real signal against a family of scaled Morlet
wavelets, computed as one `torch.fft.rfft` of the (zero-padded) signal per
trial (batched across leading dims -- channels, trials, edges, whatever the
caller stacks), one elementwise multiply per frequency against a
precomputed frequency-domain Morlet filter bank, and one batched inverse
FFT back to the time domain. This is the standard FFT-convolution-theorem
implementation of a CWT (see e.g. Torrence & Compo 1998, "A Practical Guide
to Wavelet Analysis"), parameterized here to match the specific Morlet
wavelet and log-frequency scale grid the fCWT library uses (Arts & van den
Bleek, "fCWT: An accelerated wavelet transform for scalable spectral
analyses", Nature Computational Science 2022), so its numerical output is
(near-)interchangeable with `fcwt.cwt()` -- see scripts/torch_cwt_parity.py
for the empirical validation this claim rests on.

This is an independent reimplementation derived from standard CWT/Morlet
wavelet theory plus this repo's own already-documented fcwt edge-of-support
formula (see cwt_gnn_classifiers.py's `_COI_WAVELET_FB` / `_coi_valid_mask`,
which reconstructs fcwt's cone-of-influence from that same formula because
fcwt.cwt() doesn't return one). No code from fastlib/fCWT's C++/CUDA
implementation is used or consulted here.

Wavelet: fcwt.cwt() hardcodes a Morlet wavelet with bandwidth parameter
fb=2.0 (see fcwt's own `boilerplate.cwt()`: `Morlet(2.0)`). Its edge-of-
support formula, `getSupport(scale) = int(fb*scale*3.0)` with
`scale = sampling_rate/freq` (already relied on by `_coi_valid_mask`),
implies a Gaussian-enveloped complex sinusoid whose time-domain envelope
has standard deviation `sigma_t = fb*scale` samples (3*sigma_t == that
support radius) -- i.e., in seconds, `sigma_sec = fb/freq`. That gives the
mother wavelet's frequency-domain shape, for a scale/frequency f, a
Gaussian bump centered at f with that same sigma_sec(f) controlling its
width:

    Psi_f(nu) = A * exp(-2*pi^2*sigma_sec(f)^2*(nu-f)^2)

with a PEAK AMPLITUDE A that -- confirmed empirically against real fcwt
output (see scripts/torch_cwt_parity.py), not by formula alone -- is flat
across scale/frequency, not sigma_sec(f)-dependent the way a literal
Fourier transform of a fixed-amplitude time-domain Gaussian would give.
That flat peak amplitude measures to A = sqrt(2)*pi**0.25 ~= 1.882793
(median ratio 2.50663 against real fcwt output, relative std ~0.04%, over
480 (frequency, trial) pairs across 60 real CHB-MIT trials spanning every
channel and ~all of one recording) -- close to, but not exactly, a
standard named wavelet-theory constant; treated here as an empirically
pinned-down calibration constant (fcwt's actual internal normalization
formula isn't consulted -- see this module's top docstring), not a derived
one. See MORLET_AMPLITUDE_SCALE.

Precision: fcwt.cwt() (via boilerplate.cwt()) always recasts its input to
float32 ('single') internally regardless of what dtype is passed in, so the
FFTW path this module replaces genuinely runs in float32, not float64 --
this module matches that deliberately.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# fcwt.cwt() hardcodes Morlet(2.0) -- see this module's docstring. Single
# source of truth so nothing here (or in cwt_gnn_classifiers.py's COI code,
# which independently relies on the same constant) can drift from it.
MORLET_FB = 2.0

# Peak amplitude of each filter-bank entry at its own center frequency --
# flat across scale/frequency (see module docstring). Empirically
# calibrated against real fcwt.cwt() output (scripts/torch_cwt_parity.py,
# 60 real CHB-MIT trials x 8 frequencies, ratio 2.50663 +/- 0.04% relative
# std) to <0.1% -- spelled out as its own named constant (not inlined into
# _build_filter_bank's math) so a future re-calibration is a one-line,
# visible change rather than a silent edit buried in the exponential.
MORLET_AMPLITUDE_SCALE = math.sqrt(2.0) * math.pi**0.25


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def _log_freq_grid(f0: float, f1: float, fn: int, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """fcwt's scaling="log" frequency grid: fn points geometrically spaced
    (i.e. linearly spaced in log-frequency) between f0 and f1, both
    endpoints inclusive -- the standard log-frequency CWT scale grid
    convention. Ordered index 0 -> f1 (highest), index fn-1 -> f0 (lowest)
    -- i.e. increasing SCALE / decreasing frequency, matching fcwt.cwt()'s
    own output order (confirmed empirically: scripts/torch_cwt_parity.py
    against real fcwt output; NOT derivable from the boilerplate.py source
    alone, which only shows the Scales() call, not its internal ordering).
    Getting this order right matters beyond cosmetics -- every downstream
    frequency-indexed computation (COI mask's `scale = fs/freqs`, the
    smoothing kernels, coherence/phase stacking) assumes column i of the
    coefficient array corresponds to freqs[i] at that same index."""
    if fn <= 0:
        raise ValueError("fn (nfreqs) must be a positive integer.")
    if fn == 1:
        return torch.tensor([f1], device=device, dtype=dtype)
    ratio = float(f0) / float(f1)
    exponents = torch.linspace(0.0, 1.0, fn, device=device, dtype=dtype)
    return float(f1) * ratio ** exponents


def _boundary_pad(sampling_rate: float, f0: float, fb: float = MORLET_FB) -> int:
    """Zero-padding half-width (samples) applied on each side of the signal
    before the FFT convolution, sized to the WIDEST wavelet's support (the
    lowest frequency f0 == largest scale) so that wavelet's whole envelope
    sees genuine zeros at the signal's true edges instead of wrapping
    around the FFT's circular boundary. Uses the exact same
    scale=fs/freq, support=floor(fb*scale*3.0) convention
    cwt_gnn_classifiers.py's `_coi_valid_mask` already assumes (reproduced
    there from fcwt's own edge-of-support formula) -- keeping this in
    lockstep means COI masking continues to describe this module's actual
    wavelet support, not just fcwt's, so it doesn't silently drift out of
    alignment with the new output. See that method's docstring for how the
    two are used together.
    """
    scale_max = float(sampling_rate) / float(f0)
    return int(math.ceil(fb * scale_max * 3.0))


_FILTER_BANK_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
_FILTER_BANK_CACHE_MAX_ENTRIES = 64


def _build_filter_bank(
    *,
    sampling_rate: float,
    n_time: int,
    f0: float,
    f1: float,
    fn: int,
    fb: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Builds the frequency-domain Morlet filter bank for one
    (sampling_rate, n_time, f0, f1, fn) config, once. Returns
    (freqs [fn] float32, filters [fn, n_fft_bins] complex64, pad, n_padded),
    already moved to `device`.

    Always built in float64 on CPU, regardless of `device` -- MPS doesn't
    support float64 at all (a fused `.to(device, dtype=torch.float64)`
    raises there), and this exponential is numerically sensitive enough at
    the widest/narrowest scales to want the precision. Same two-step
    "build in float64 on CPU, then move" convention
    cwt_gnn_classifiers.py's `_scale_adaptive_time_kernel` already uses for
    the same reason. Cheap either way -- this only runs once per config,
    not per call (see _get_filter_bank's cache).
    """
    pad = _boundary_pad(sampling_rate, f0, fb=fb)
    n_padded = _next_pow2(n_time + 2 * pad)

    freqs64 = _log_freq_grid(float(f0), float(f1), int(fn), device="cpu", dtype=torch.float64)
    sigma_sec = fb / freqs64  # seconds -- see module docstring's derivation

    n_fft_bins = n_padded // 2 + 1
    bin_freqs = torch.fft.rfftfreq(n_padded, d=1.0 / float(sampling_rate)).to(
        dtype=torch.float64
    )  # Hz, [n_fft_bins], CPU

    delta = bin_freqs.unsqueeze(0) - freqs64.unsqueeze(1)  # [fn, n_fft_bins], Hz
    # Peak amplitude MORLET_AMPLITUDE_SCALE is flat across frequency -- see
    # module docstring; sigma_sec only shapes each bump's WIDTH here, not
    # its height.
    filters64 = MORLET_AMPLITUDE_SCALE * torch.exp(
        -2.0 * (math.pi**2) * (sigma_sec.unsqueeze(1) ** 2) * (delta**2)
    )  # [fn, n_fft_bins], real-valued Gaussian bumps, CPU float64

    freqs = freqs64.to(torch.float32).to(device)
    filters = filters64.to(torch.complex64).to(device)  # real filter, complex dtype for the spectrum multiply
    return freqs, filters, pad, n_padded


def _get_filter_bank(
    *, sampling_rate: float, n_time: int, f0: float, f1: float, fn: int, fb: float, device
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    key = (
        float(sampling_rate),
        int(n_time),
        float(f0),
        float(f1),
        int(fn),
        float(fb),
        str(device),
    )
    cached = _FILTER_BANK_CACHE.get(key)
    if cached is not None:
        return cached
    result = _build_filter_bank(
        sampling_rate=sampling_rate, n_time=n_time, f0=f0, f1=f1, fn=fn, fb=fb, device=device
    )
    if len(_FILTER_BANK_CACHE) >= _FILTER_BANK_CACHE_MAX_ENTRIES:
        # Simple bound, not an LRU: configs are few and stable within a run
        # (one per distinct window length/CWT config), so this only guards
        # against unbounded growth if something calls this with many
        # distinct n_time values -- not expected in practice.
        _FILTER_BANK_CACHE.clear()
    _FILTER_BANK_CACHE[key] = result
    return result


def cwt_torch(
    signal: torch.Tensor,
    sampling_rate: float,
    f0: float,
    f1: float,
    fn: int,
    *,
    fb: float = MORLET_FB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Morlet CWT via torch.fft.

    Parameters
    ----------
    signal : torch.Tensor, real, shape [..., T]
        Any number of leading batch dims (e.g. [n_channels, T] or
        [n_samples, n_channels, T]) -- every leading-dim slice is
        transformed independently (no cross-slice mixing), so this is
        exactly equivalent to calling this once per slice, just batched.
    sampling_rate : Hz.
    f0, f1 : lowest/highest frequency, Hz (f0 < f1).
    fn : number of log-spaced frequencies.

    Returns
    -------
    coeffs : torch.Tensor, complex64, shape [..., fn, T]
        Freq-major, matching fcwt.cwt()'s own (nfreqs, n_time) convention.
    freqs : torch.Tensor, float32, shape [fn]
    """
    if signal.ndim < 1:
        raise ValueError("signal must have at least one (time) dimension.")
    if f0 <= 0 or f1 <= f0:
        raise ValueError(f"Require 0 < f0 < f1, got f0={f0}, f1={f1}.")

    n_time = int(signal.shape[-1])
    freqs, filters, pad, n_padded = _get_filter_bank(
        sampling_rate=sampling_rate, n_time=n_time, f0=f0, f1=f1, fn=fn, fb=fb, device=signal.device
    )

    x = signal.to(torch.float32)
    x_padded = torch.nn.functional.pad(x, (pad, n_padded - n_time - pad))
    spectrum = torch.fft.rfft(x_padded, n=n_padded, dim=-1)  # [..., n_fft_bins] complex64

    product = spectrum.unsqueeze(-2) * filters  # [..., fn, n_fft_bins] complex64

    # torch.fft.irfft can't be used for the inverse step: it always
    # reconstructs a REAL signal (it assumes Hermitian/conjugate symmetry),
    # but the Morlet filter is one-sided/analytic by construction (zero for
    # negative frequencies) -- that's exactly what makes the CWT
    # coefficient complex (magnitude + phase) in the first place. So invert
    # via a full complex ifft on the product zero-extended over the
    # negative-frequency half (NOT Hermitian-mirrored, genuinely zero) --
    # the standard analytic-signal reconstruction for an FFT-based CWT.
    n_fft_bins = n_padded // 2 + 1
    full_spectrum = torch.nn.functional.pad(product, (0, n_padded - n_fft_bins))
    coeffs_padded = torch.fft.ifft(full_spectrum, n=n_padded, dim=-1)  # [..., fn, n_padded] complex64
    coeffs = coeffs_padded[..., pad : pad + n_time].contiguous()

    return coeffs.to(torch.complex64), freqs


def transform(
    signal1: np.ndarray,
    frame_rate: float,
    highest: float,
    lowest: float,
    nfreqs: int = 100,
    *,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in-signature replacement for `utils.coherence_utils.transform`
    (same positional args, same (coeffs, freqs) return order/shape/dtype),
    backed by `cwt_torch` (torch.fft) instead of fcwt/FFTW. See this
    module's top docstring for the algorithm and the precision note.

    `device`, if given, runs the transform on that torch device (CPU by
    default, matching `numpy` in/out) -- see scripts/torch_cwt_parity.py's
    CPU-vs-device parity check before trusting a non-CPU device here.
    """
    signal1 = np.asarray(signal1)
    if signal1.ndim != 1:
        raise ValueError("signal1 must be a 1-D array.")

    x = torch.from_numpy(np.ascontiguousarray(signal1, dtype=np.float32))
    if device is not None:
        x = x.to(device)

    coeffs_t, freqs_t = cwt_torch(x, sampling_rate=frame_rate, f0=lowest, f1=highest, fn=nfreqs)

    coeffs = coeffs_t.detach().cpu().numpy().astype(np.complex64)
    freqs = freqs_t.detach().cpu().numpy().astype(np.float32)
    return coeffs, freqs


def transform_batch(
    signals: np.ndarray,
    frame_rate: float,
    highest: float,
    lowest: float,
    nfreqs: int = 100,
    *,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched sibling of `transform`. Same params, except `signals` is
    2-D, [n_signals, T] -- independent 1-D windows/channels stacked along a
    new leading axis (e.g. every cache-miss (sample, channel) pair from one
    _prepare_features call) -- instead of a single 1-D array.

    Exists because `cwt_torch` is already batched over arbitrary leading
    dims, but nothing calling plain `transform()` in a Python loop (the
    call sites in cwt_window_cache.py / common.py, written against
    fcwt.cwt()'s inherently single-signal interface) was exercising that.
    Looping `transform()` with `device="cuda"` there would do one
    host<->device transfer plus one tiny kernel launch per signal --
    exactly the overhead-bound regime that erases torch_cwt's measured
    speedup (see Epilepsy/Session_notes/2026_08_20/
    torch_native_cwt_module_and_parity_validation.md, Part 8/9). This is
    the one-call-for-the-whole-batch alternative those call sites use
    instead when the active transform backend is torch_cwt.

    Returns (coeffs [n_signals, nfreqs, T] complex64, freqs [nfreqs] float32)
    -- same freqs for every signal in the batch (frame_rate/highest/lowest/
    nfreqs are shared config, not per-signal), matching what those call
    sites already assume (they only ever read freqs from an [0, 0]-index
    probe call).
    """
    signals = np.asarray(signals)
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2-D, [n_signals, n_time]; got shape {signals.shape}.")

    x = torch.from_numpy(np.ascontiguousarray(signals, dtype=np.float32))
    if device is not None:
        x = x.to(device)

    coeffs_t, freqs_t = cwt_torch(x, sampling_rate=frame_rate, f0=lowest, f1=highest, fn=nfreqs)

    coeffs = coeffs_t.detach().cpu().numpy().astype(np.complex64)
    freqs = freqs_t.detach().cpu().numpy().astype(np.float32)
    return coeffs, freqs