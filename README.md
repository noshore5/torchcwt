# torchcwt

**PyTorch-native continuous wavelet transform using Morlet wavelets and `torch.fft`.**

`torchcwt` is a from-scratch implementation of the continuous wavelet transform designed to reproduce the numerical behavior of [`fCWT`](https://github.com/fastlib/fCWT) without depending on its Python package, C++ implementation, or FFTW backend.

The implementation is written entirely in PyTorch and supports CPU, CUDA, and MPS execution. The core transform accepts arbitrary leading batch dimensions, making it suitable for processing many signals in a single call.

The implementation targets **near-interchangeability with `fcwt.cwt()`** in coefficient magnitude, phase, frequency grid, and output convention. This parity has been empirically validated against real fCWT output.

## Features

* **Pure PyTorch** — the transform is implemented with `torch.fft` and PyTorch tensor operations.
* **CPU, CUDA, and MPS** — the transform runs on the device of the input tensor.
* **Arbitrary batching** — inputs can have any number of leading dimensions: `[..., T]`.
* **Morlet wavelet** — uses `fb=2.0`, matching the Morlet wavelet used by `fcwt.cwt()`.
* **Log-spaced frequencies** — frequencies are ordered from highest to lowest, matching fCWT.
* **Frequency-domain filter bank** — Morlet filters are constructed in the frequency domain and cached by configuration.
* **NumPy wrappers** — `transform()` and `transform_batch()` provide interfaces compatible with common fCWT-based pipelines.
* **Complex coefficients** — returns magnitude and phase through complex-valued CWT coefficients.
* **Support-aware padding** — zero-padding is sized according to the support of the widest wavelet to prevent circular FFT convolution from wrapping the wavelet around the signal boundaries.

## Installation

### From PyPI

```bash
pip install torchcwt
```

### From GitHub

```bash
pip install "git+https://github.com/noshore5/torchcwt.git"
```

To install a specific release:

```bash
pip install "git+https://github.com/noshore5/torchcwt.git@v0.1.0"
```

### Editable installation

```bash
git clone https://github.com/noshore5/torchcwt.git
cd torchcwt
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick start

### PyTorch

The native PyTorch interface is the preferred interface when the surrounding pipeline already uses PyTorch.

```python
import torch
from torchcwt import cwt_torch

x = torch.randn(8, 4, 4096, device="cuda")

coeffs, freqs = cwt_torch(
    x,
    sampling_rate=256.0,
    f0=1.0,
    f1=40.0,
    fn=32,
)
```

The final dimension is interpreted as time. All preceding dimensions are treated independently and processed as a batch:

```text
Input:   [8, 4, 4096]
Output:  [8, 4, 32, 4096]
```

where:

* `8` = batch dimension
* `4` = channels
* `32` = frequency bins
* `4096` = time samples

The returned coefficients are `complex64` and the frequencies are `float32` on the same device as the input.

### NumPy

For pipelines that use NumPy arrays, `transform()` provides a simple wrapper around the PyTorch implementation:

```python
import numpy as np
from torchcwt import transform

signal = np.random.randn(4096).astype(np.float32)

coeffs, freqs = transform(
    signal,
    frame_rate=256.0,
    highest=40.0,
    lowest=1.0,
    nfreqs=32,
)

print(coeffs.shape)
# (32, 4096)

print(freqs.shape)
# (32,)
```

The frequency axis is ordered from highest frequency to lowest frequency.

### NumPy batch interface

Multiple signals can be transformed in one call:

```python
import numpy as np
from torchcwt import transform_batch

signals = np.random.randn(16, 4096).astype(np.float32)

coeffs, freqs = transform_batch(
    signals,
    frame_rate=256.0,
    highest=40.0,
    lowest=1.0,
    nfreqs=32,
    device="cuda",
)

print(coeffs.shape)
# (16, 32, 4096)
```

This avoids repeatedly transferring individual signals to the GPU and launching a separate transform for each one.

## API

### `cwt_torch`

```python
cwt_torch(
    signal,
    sampling_rate,
    f0,
    f1,
    fn,
    *,
    fb=2.0,
)
```

Core batched CWT.

#### Parameters

| Parameter       | Description                                      |
| --------------- | ------------------------------------------------ |
| `signal`        | Real PyTorch tensor with shape `[..., T]`.       |
| `sampling_rate` | Sampling frequency in Hz.                        |
| `f0`            | Lowest frequency in Hz.                          |
| `f1`            | Highest frequency in Hz.                         |
| `fn`            | Number of logarithmically spaced frequency bins. |
| `fb`            | Morlet bandwidth parameter. Defaults to `2.0`.   |

#### Returns

```text
coeffs
    shape: [..., fn, T]
    dtype: complex64

freqs
    shape: [fn]
    dtype: float32
```

Every leading-dimension slice is transformed independently. No information is mixed between batch dimensions.

---

### `transform`

```python
transform(
    signal1,
    frame_rate,
    highest,
    lowest,
    nfreqs=100,
    *,
    device=None,
)
```

NumPy wrapper for a single one-dimensional signal.

The signature and return convention are intended to be compatible with existing fCWT-based code:

```text
signal1
    [T]

coeffs
    [nfreqs, T] complex64

freqs
    [nfreqs] float32
```

By default the transform runs on CPU. Passing `device="cuda"` or another PyTorch device moves the computation to that device before returning NumPy arrays.

---

### `transform_batch`

```python
transform_batch(
    signals,
    frame_rate,
    highest,
    lowest,
    nfreqs=100,
    *,
    device=None,
)
```

Batched NumPy wrapper.

Input:

```text
[n_signals, T]
```

Output:

```text
coeffs: [n_signals, nfreqs, T]
freqs:  [nfreqs]
```

All signals share the same sampling rate and frequency configuration.

---

### Constants

```python
from torchcwt import MORLET_FB
```

`MORLET_FB` is the Morlet bandwidth parameter used by the implementation:

```python
MORLET_FB == 2.0
```

This matches the `Morlet(2.0)` parameterization used by `fcwt.cwt()`.

`MORLET_AMPLITUDE_SCALE` is the empirically calibrated peak amplitude used for the frequency-domain filter bank. The value was calibrated against real fCWT output rather than derived solely from the standard wavelet normalization.

## How it works

The transform uses the Fourier convolution theorem.

For an input signal `x(t)` and a family of scaled Morlet wavelets, convolution in the time domain becomes multiplication in the frequency domain.

`torchcwt` therefore:

1. Determines the logarithmically spaced frequency grid.
2. Determines the support of the widest wavelet.
3. Zero-pads the signal on both sides.
4. Pads the total signal length to the next power of two.
5. Computes one `torch.fft.rfft()` along the time dimension.
6. Multiplies the spectrum by a precomputed frequency-domain Morlet filter bank.
7. Reconstructs the complex analytic coefficients using a full `torch.fft.ifft()`.
8. Removes the padding and returns the original time length.

The filter bank is cached by transform configuration, so repeated transforms with the same sampling rate, window length, frequency range, number of frequencies, wavelet parameter, and device reuse the existing filters.

### Frequency grid

The frequency grid is logarithmically spaced between `f1` and `f0`:

```text
f1 → f0
highest → lowest
```

The first frequency bin corresponds to the highest requested frequency, while the final bin corresponds to the lowest. This ordering matches the output convention of `fcwt.cwt()`.

### Morlet filters

The frequency-domain filters are Gaussian functions centered on each requested frequency.

For frequency `f`, the implementation uses

```text
σ = fb / f
```

in seconds, giving a frequency-domain Gaussian of the form

```text
Ψ_f(ν) = A exp[-2π² σ² (ν - f)²]
```

where `A` is the empirically calibrated `MORLET_AMPLITUDE_SCALE`.

The filters are constructed in `float64` on the CPU and then converted to `complex64` and moved to the requested device. This avoids numerical issues in the filter construction while remaining compatible with MPS, which does not support `float64` tensors on-device.

### Boundary handling

The transform uses zero-padding before performing the FFT convolution.

The padding is based on the support of the widest wavelet, corresponding to the lowest requested frequency:

```text
scale = sampling_rate / f0
support = floor(fb * scale * 3)
```

The implementation uses sufficient padding on each side so that the wavelet envelope encounters zeros rather than wrapping around the signal through the FFT's circular boundary.

**`torchcwt` does not calculate or return a cone-of-influence mask.** The support calculation is used for boundary padding; it should not be interpreted as the package providing a COI output.

## Relationship to fCWT

`torchcwt` was developed specifically to replace an FFTW-backed `fcwt.cwt()` call in a PyTorch-based signal-processing pipeline.

It is an **independent implementation**. It does not use code from the fCWT C++/CUDA implementation. Instead, it reproduces the relevant behavior from standard CWT/Morlet theory and empirically validated fCWT conventions.

The implementation matches important aspects of fCWT including:

* Morlet `fb=2.0`
* logarithmic frequency spacing
* frequency ordering
* float32 signal processing
* complex coefficient output
* frequency-domain normalization
* wavelet support convention

The numerical parity was tested directly against real `fcwt.cwt()` output across real signals and frequencies. The empirical calibration of the filter amplitude is documented in `cwt.py`.

The goal is therefore not merely to provide another CWT implementation, but to provide a **PyTorch-native implementation whose output can be substituted into existing fCWT-based pipelines with minimal changes**.

## Performance and batching

The main motivation for the implementation is to keep the entire transform in PyTorch so that it can participate directly in GPU-based pipelines.

A single batched FFT is performed over all leading dimensions of the input, rather than launching a separate Python-level transform for every signal. The same filter bank is then applied across the batch.

For example:

```python
x.shape
# [batch, channels, time]

coeffs, freqs = cwt_torch(
    x,
    sampling_rate=256,
    f0=1,
    f1=40,
    fn=32,
)

coeffs.shape
# [batch, channels, 32, time]
```

For NumPy pipelines, `transform_batch()` exposes the same batching behavior without requiring the caller to manually convert arrays to PyTorch tensors.

## Requirements

* Python ≥ 3.9
* NumPy ≥ 1.20
* PyTorch ≥ 1.12

PyTorch should be installed with the appropriate backend for the hardware being used.

## Development

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/noshore5/torchcwt.git
cd torchcwt
pip install -e ".[dev]"
```

Run the test suite with:

```bash
pytest
```

The repository also contains the parity-validation work used to compare `torchcwt` against fCWT.

## References

### fCWT

Arts, R. & van den Bleek, C.
**fCWT: An accelerated wavelet transform for scalable spectral analyses.**
*Nature Computational Science* (2022).

The fCWT project and implementation are available at:

https://github.com/fastlib/fCWT

### Continuous wavelet transform

Torrence, C. & Compo, G. P. (1998).
**A Practical Guide to Wavelet Analysis.**
*Bulletin of the American Meteorological Society*, 79(1), 61–78.

DOI: https://doi.org/10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2

## Acknowledgements

`torchcwt` is an independent implementation, but it was developed to reproduce the numerical behavior and conventions of fCWT.

Credit goes to the authors and contributors of **fCWT** for the original implementation and methodology that this project targets for compatibility.

`torchcwt` is not affiliated with, endorsed by, or an official implementation of fCWT.

## License

MIT License.
