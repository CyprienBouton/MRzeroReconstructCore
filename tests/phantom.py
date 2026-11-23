"""Synthetic multi-coil phantom shared by the tests (no data files needed).

This file is identical in MRzeroReconstructCore and MRTwixReconstructCore.
"""

import math

import torch


def make_phantom(shape=(32, 32, 4), ncoils=4, smooth_phase=True, seed=0):
    """Return ``(coil_images, kspace)`` of a smooth multi-coil object.

    * object: ellipsoid with a smooth (low-order) phase,
    * coils: Gaussian sensitivities placed around the object,
    * kspace: centred unitary FFT of the coil images, ``[nx, ny, nz, nc]``.
    """
    reco_utils = _reco_utils()
    generator = torch.Generator().manual_seed(seed)
    nx, ny, nz = shape
    x = torch.linspace(-1, 1, nx).view(nx, 1, 1)
    y = torch.linspace(-1, 1, ny).view(1, ny, 1)
    # Slightly off-centre z grid: a symmetric profile would have an exactly
    # zero Nyquist (kz) sample, which looks "unacquired" to the mask helpers.
    z = torch.linspace(-0.9, 1.1, nz).view(1, 1, nz) if nz > 1 else torch.zeros(1, 1, 1)
    obj = ((x / 0.7) ** 2 + (y / 0.6) ** 2 + (z / 1.2) ** 2 <= 1).to(torch.float32)
    obj = obj * (1.0 + 0.3 * torch.cos(3 * x) * torch.sin(2 * y))
    phase = 0.6 * x + 0.4 * y + 0.3 * x * y if smooth_phase else torch.zeros_like(obj)
    obj = obj * torch.exp(1j * phase)

    coils = []
    for c in range(ncoils):
        angle = 2 * math.pi * c / ncoils
        cx, cy = 1.2 * math.cos(angle), 1.2 * math.sin(angle)
        sens = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 1.5)
        sens = sens * torch.exp(1j * (0.5 * c + 0.2 * torch.randn(1, generator=generator)))
        coils.append(obj * sens * torch.ones_like(z))
    images = torch.stack(coils, dim=-1).to(torch.complex64)
    kspace = reco_utils.centered_fft(images, dims=(0, 1, 2))
    return images, kspace


def relative_error(estimate, reference):
    return float(torch.linalg.norm(estimate - reference) / torch.linalg.norm(reference))


def _reco_utils():
    from importlib import import_module

    for name in ("MRzeroReconstructCoreLib", "MRTwinReconstructCoreLib"):
        try:
            return import_module(name + ".reco_utils")
        except ImportError:
            continue
    raise ImportError("No reconstruction core package found")
