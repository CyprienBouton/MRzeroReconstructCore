"""Wavelet-regularized SENSE reconstruction on a centered Cartesian grid."""

import math
import sys

import torch

from .reco_utils import centered_fft, centered_ifft, rss


def _estimate_sensitivities(kspace, acquired, widths):
    """Normalize low-frequency coil images to unit root-sum-of-squares."""
    if kspace.shape[-1] == 1:
        return torch.ones_like(kspace)

    shape = kspace.shape[:3]
    window = []
    for size, width in zip(shape, widths):
        width = min(size, width)
        start = size // 2 - width // 2
        window.append(slice(start, start + width))
    window = tuple(window)
    calibration = torch.zeros_like(acquired)
    calibration[window] = acquired[window]
    if not calibration.any():
        raise ValueError("The central calibration window contains no acquired samples.")

    if sys.version_info < (3, 10):
        # Slicer's Python 3.9 can evaluate the same Cartesian adjoint via FFT.
        coil_images = centered_ifft(kspace.masked_fill(~calibration[..., None], 0))
    else:
        from mrinufft import get_operator

        # MRI-NUFFT takes coordinates in cycles per voxel and coil-first data.
        indices = calibration.nonzero()
        sizes = indices.new_tensor(shape)
        trajectory = ((indices - sizes // 2).to(kspace.real.dtype) / sizes).cpu().numpy()
        samples = kspace[calibration].T.contiguous().detach().cpu().numpy()
        operator = get_operator("finufft")(
            trajectory, tuple(shape), n_coils=kspace.shape[-1], nthreads=1,
        )
        coil_images = torch.as_tensor(
            operator.adj_op(samples), dtype=kspace.dtype, device=kspace.device,
        ).reshape(kspace.shape[-1], *shape).movedim(0, -1)

    magnitude = rss(coil_images)[..., None]
    floor = (magnitude.max() * torch.finfo(magnitude.dtype).eps).clamp_min(
        torch.finfo(magnitude.dtype).tiny
    )
    return coil_images / magnitude.clamp_min(floor)


def _haar_axis(data, axis, inverse):
    """Orthonormal Haar step, carrying an unpaired endpoint into the low band."""
    data = data.movedim(axis, 0)
    pairs = data.shape[0] // 2
    if pairs == 0:
        return data.movedim(0, axis)
    low_size = data.shape[0] - pairs
    if inverse:
        low, high = data[:pairs], data[low_size:]
        even = (low + high) / math.sqrt(2)
        odd = (low - high) / math.sqrt(2)
        paired = torch.stack((even, odd), dim=1).flatten(0, 1)
        result = torch.cat((paired, data[pairs:low_size]))
    else:
        even, odd = data[:2 * pairs:2], data[1:2 * pairs:2]
        low = (even + odd) / math.sqrt(2)
        high = (even - odd) / math.sqrt(2)
        result = torch.cat((low, data[2 * pairs:], high))
    return result.movedim(0, axis)


def _wavelet_shrink(image, threshold, levels):
    """Apply complex soft thresholding in an orthonormal Haar basis."""
    coefficients = image.clone()
    for region in levels:
        band = coefficients[region]
        for axis in range(image.ndim):
            band = _haar_axis(band, axis, inverse=False)
        coefficients[region] = band

    magnitude = coefficients.abs()
    coefficients *= (magnitude - threshold).clamp_min(0) / magnitude.clamp_min(
        torch.finfo(magnitude.dtype).tiny
    )

    for region in reversed(levels):
        band = coefficients[region]
        for axis in reversed(range(image.ndim)):
            band = _haar_axis(band, axis, inverse=True)
        coefficients[region] = band
    return coefficients


@torch.no_grad()
def reconstruct_cartesian_cs(kspace, acquired, retained, regularization, calib_kernel):
    """Solve masked SENSE with L1 wavelet regularization using FISTA."""
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    widths = tuple(int(width) for width in calib_kernel.split(":"))
    if len(widths) != 3 or min(widths) <= 0:
        raise ValueError("calib_kernel must contain three positive widths (x:y:z).")

    mask = retained[..., None]
    measured = kspace.masked_fill(~mask, 0)
    scale = measured.abs().max()
    if scale == 0:
        return torch.zeros_like(kspace[..., 0])
    measured /= scale

    sensitivities = _estimate_sensitivities(kspace, acquired, widths)
    conjugate_maps = sensitivities.conj()

    def adjoint(data):
        return (centered_ifft(data) * conjugate_maps).sum(dim=-1)

    levels = []
    shape = kspace.shape[:3]
    while max(shape) > 1:
        levels.append(tuple(slice(n) for n in shape))
        shape = tuple((n + 1) // 2 for n in shape)

    # The unitary FFT, sample selection and normalized maps give a Lipschitz
    # bound of one for the gradient, so a unit step is valid.
    image = extrapolated = adjoint(measured)
    momentum = 1.0
    for _ in range(50):
        predicted = centered_fft(extrapolated[..., None] * sensitivities)
        gradient = adjoint((predicted - measured).masked_fill(~mask, 0))
        updated = extrapolated - gradient
        if regularization:
            updated = _wavelet_shrink(updated, regularization, levels)
        next_momentum = (1 + math.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - image)
        image, momentum = updated, next_momentum

    return image
