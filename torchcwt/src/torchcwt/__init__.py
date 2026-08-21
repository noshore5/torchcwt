"""Torch-native continuous wavelet transform (Morlet).

A from-scratch, torch-only numerical replacement for fCWT / FFTW-backed
Morlet CWT, implemented via the Fourier convolution theorem in torch.fft.
"""

from .cwt import (
    MORLET_AMPLITUDE_SCALE,
    MORLET_FB,
    cwt_torch,
    transform,
    transform_batch,
)

__all__ = [
    "MORLET_FB",
    "MORLET_AMPLITUDE_SCALE",
    "cwt_torch",
    "transform",
    "transform_batch",
]

__version__ = "0.1.0"
