"""Shared Cartesian reconstruction utilities.

Array conventions
-----------------
* Multi-coil k-space is a complex ``torch.Tensor`` of shape
  ``[nx, ny, nz, ncoils]``: readout (Col), phase encoding (Lin),
  partition / slice encoding (Par), coil (Cha).
* Unacquired k-space samples are exactly zero in every coil.
* A k-space array is *centred* when the k = 0 sample sits at index ``n // 2``
  of each encoding axis. This is the convention of
  ``torch.fft.ifftshift`` / ``fftshift``, so a centred array can be passed
  directly to :func:`centered_ifft`.
"""

from collections import Counter
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch

ENCODING_DIMS = (0, 1, 2)
IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# =====================================================================
# Centred Fourier transforms
# =====================================================================

def centered_ifft(data: torch.Tensor, dims: Sequence[int] = ENCODING_DIMS,
                  norm: str = "ortho") -> torch.Tensor:
    """Centred inverse FFT (k-space -> image) over ``dims``.

    Computes ``fftshift(ifftn(ifftshift(data)))``: the input has k = 0 at
    index ``n // 2`` and the output has the image centre at index ``n // 2``.
    With ``norm="ortho"`` the transform is unitary.
    """
    dims = tuple(sorted(set(dims)))
    if not dims:
        return data
    data = torch.fft.ifftshift(data, dim=dims)
    data = torch.fft.ifftn(data, dim=dims, norm=norm)
    return torch.fft.fftshift(data, dim=dims)


def centered_fft(data: torch.Tensor, dims: Sequence[int] = ENCODING_DIMS,
                 norm: str = "ortho") -> torch.Tensor:
    """Centred forward FFT (image -> k-space); inverse of :func:`centered_ifft`."""
    dims = tuple(sorted(set(dims)))
    if not dims:
        return data
    data = torch.fft.ifftshift(data, dim=dims)
    data = torch.fft.fftn(data, dim=dims, norm=norm)
    return torch.fft.fftshift(data, dim=dims)


def rss(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Root-sum-of-squares coil combination: ``sqrt(sum_c |x_c|^2)``."""
    return torch.sqrt(torch.sum(torch.abs(data) ** 2, dim=dim))


# =====================================================================
# Centring by zero-padding
# =====================================================================

def get_centering_padding(shape: Sequence[int], center: Sequence[int]):
    """Zero-padding that moves ``center`` to index ``n // 2`` on each axis.

    Parameters
    ----------
    shape:
        Array shape, e.g. ``[nx, ny, nz, ncoils]``.
    center:
        Index of the k = 0 sample along the first ``len(center)`` axes.

    Returns
    -------
    padding:
        ``[(before, after), ...]`` for every axis of ``shape`` (axes beyond
        ``len(center)`` get ``(0, 0)``), usable with ``np.pad``.
    new_center:
        Index of k = 0 in the padded array (always ``n_padded // 2``).

    Notes
    -----
    Samples are never discarded or wrapped. An axis that is already centred
    (``center == n // 2``) is left untouched. Otherwise, e.g. for partial
    Fourier / asymmetric echo, the axis is padded to the smallest even size
    ``2 * L`` with ``L = max(c, n - c)``; k = 0 then sits at index ``L``.
    """
    padding = []
    new_center = []
    for axis, c in enumerate(center):
        if c is None:
            raise ValueError(f"k-space centre for axis {axis} is undefined.")
        c = int(c)
        n = int(shape[axis])
        if not 0 <= c < n:
            raise ValueError(
                f"k-space centre {c} is outside axis {axis} of size {n}."
            )
        if c == n // 2:
            before = after = 0
            new_c = c
        else:
            half = max(c, n - c)
            before = half - c
            after = 2 * half - n - before
            new_c = half
        padding.append((before, after))
        new_center.append(new_c)
    padding.extend([(0, 0)] * (len(shape) - len(center)))
    return padding, tuple(new_center)


def center_kspace(kspace: torch.Tensor, center: Sequence[int]):
    """Zero-pad ``kspace`` so that ``center`` lands at index ``n // 2``.

    The input tensor is not modified. See :func:`get_centering_padding`.

    Returns
    -------
    (centered_kspace, new_center)
    """
    padding, new_center = get_centering_padding(kspace.shape, center)
    if all(before == 0 and after == 0 for before, after in padding):
        return kspace.clone(), new_center
    new_shape = [n + before + after
                 for n, (before, after) in zip(kspace.shape, padding)]
    centered = torch.zeros(new_shape, dtype=kspace.dtype, device=kspace.device)
    slices = tuple(slice(before, before + n)
                   for n, (before, _) in zip(kspace.shape, padding))
    centered[slices] = kspace
    return centered, new_center


# =====================================================================
# Sampling pattern, ACS region and acceleration factor
# =====================================================================

def get_sampling_mask(kspace: torch.Tensor) -> torch.Tensor:
    """Boolean ``[nx, ny, nz]`` mask of acquired positions (non-zero in any coil)."""
    return kspace.abs().sum(dim=-1) > 0


def get_acs_bounds(kspace: torch.Tensor,
                   center: Optional[Sequence[int]] = None):
    """Bounds of the largest fully sampled box around the k-space centre.

    Starting from the single voxel at ``center`` (default ``n // 2`` on each
    axis), the box is grown greedily one line at a time on each side of each
    encoding axis for as long as the newly added slab is fully sampled.

    Returns
    -------
    ((x0, x1), (y0, y1), (z0, z1)) with exclusive upper bounds.
    """
    mask = get_sampling_mask(kspace)
    shape = tuple(mask.shape)
    if center is None:
        center = [n // 2 for n in shape]
    lo = [int(c) for c in center]
    hi = [int(c) + 1 for c in center]
    if not bool(mask[tuple(lo)]):
        raise ValueError(
            "The k-space centre is not sampled; cannot locate the ACS region."
        )
    grown = True
    while grown:
        grown = False
        for axis in range(3):
            for side in (-1, 1):
                if side < 0 and lo[axis] > 0:
                    index = lo[axis] - 1
                elif side > 0 and hi[axis] < shape[axis]:
                    index = hi[axis]
                else:
                    continue
                slab = [slice(lo[a], hi[a]) for a in range(3)]
                slab[axis] = slice(index, index + 1)
                if bool(mask[tuple(slab)].all()):
                    if side < 0:
                        lo[axis] -= 1
                    else:
                        hi[axis] += 1
                    grown = True
    return tuple(zip(lo, hi))


def get_acs_from_kspace(kspace: torch.Tensor,
                        center: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Extract the fully sampled auto-calibration (ACS) block.

    Returns a view ``[n_acs_x, n_acs_y, n_acs_z, ncoils]`` of ``kspace``;
    see :func:`get_acs_bounds`.
    """
    (x0, x1), (y0, y1), (z0, z1) = get_acs_bounds(kspace, center)
    return kspace[x0:x1, y0:y1, z0:z1, :]


def estimate_acceleration(kspace: torch.Tensor) -> Tuple[int, int]:
    """Estimate the parallel-imaging acceleration ``(af_y, af_z)``.

    For the phase (y) and partition (z) axes, the acquired line indices are
    collected and the most frequent gap larger than one is taken as the
    acceleration factor (1 when there is no gap). The fully sampled ACS
    block contributes gaps of 1 and is therefore ignored. This assumes a
    regular Cartesian pattern (no CAIPIRINHA shift, no Poisson-disc
    sampling).
    """
    mask = get_sampling_mask(kspace)
    factors = []
    for other_axes in ((0, 2), (0, 1)):
        lines = torch.where(mask.any(dim=other_axes))[0]
        gaps = [int(g) for g in torch.diff(lines).tolist() if g > 1]
        factors.append(Counter(gaps).most_common(1)[0][0] if gaps else 1)
    return factors[0], factors[1]


# =====================================================================
# GRAPPA
# =====================================================================

def grappa_reconstruct(kspace: torch.Tensor, acs: torch.Tensor,
                       af: Optional[Sequence[int]] = None,
                       kernel_size: Sequence[int] = (4, 4, 5),
                       reg: float = 1e-4, batch_size: int = 1,
                       cuda: Optional[bool] = None,
                       quiet: bool = True) -> torch.Tensor:
    """Fill the missing k-space lines with GRAPPA (vendored ``ggrappa``).

    GRAPPA assumes that every missing sample of coil ``j`` is a fixed linear
    combination of the acquired neighbouring samples of *all* coils::

        s_j(k) = sum_c sum_{d in kernel} w_{j,c,d} * s_c(k + d)

    The complex weights ``w`` are shift-invariant. They are calibrated on the
    fully sampled ACS block, where both the source pattern ``S`` (acquired
    neighbours, as they would appear with acceleration ``af``) and the
    targets ``T`` are known, by Tikhonov-regularised least squares::

        W = (S^H S + lambda^2 * sigma_max * I)^(-1) S^H T

    with ``lambda = reg`` and ``sigma_max`` the largest eigenvalue of
    ``S^H S`` (ggrappa's ``pinv``). The kernel is then slid over the
    undersampled k-space; acquired samples are kept unchanged.

    Parameters
    ----------
    kspace:
        Undersampled k-space ``[nx, ny, nz, ncoils]``.
    acs:
        Fully sampled calibration block ``[ax, ay, az, ncoils]``.
    af:
        Acceleration ``(af_y, af_z)``. ``None`` estimates it with
        :func:`estimate_acceleration`.
    kernel_size:
        Kernel extent in *acquired* lines ``(ky, kz, kx)`` (ggrappa order).
        It is clipped to the ACS size along each axis.
    reg:
        Tikhonov regularisation ``lambda`` (relative to the largest singular
        value of the calibration matrix).
    batch_size:
        Number of kernel positions processed per step (memory vs speed).
    cuda:
        Run on the GPU. ``None`` uses the GPU only when CUDA is available.
    quiet:
        Disable ggrappa logging and progress bars.

    Returns
    -------
    torch.Tensor
        Filled k-space ``[nx, ny, nz, ncoils]`` (complex64, on the CPU).
    """
    from . import ggrappa  # vendored, see ggrappa/LICENSE

    if af is None:
        af = estimate_acceleration(kspace)
    af = [int(a) for a in af]
    if all(a == 1 for a in af):
        return kspace.clone()  # nothing to fill
    if cuda is None:
        cuda = torch.cuda.is_available()

    # ggrappa expects [ncoils, ky, kz, kx].
    sig = kspace.permute(3, 1, 2, 0).contiguous()
    acs_g = acs.permute(3, 1, 2, 0).contiguous()

    try:
        filled, _ = ggrappa.GRAPPA_Recon(
            sig=sig,
            acs=acs_g,
            af=af,
            kernel_size=tuple(kernel_size),
            lambda_=reg,
            batch_size=batch_size,
            cuda=cuda,
            quiet=quiet,
        )
    except RuntimeError as exc:
        if "cannot be multiplied" in str(exc):
            # ggrappa's pinv only supports over-determined calibrations.
            raise ValueError(
                f"GRAPPA calibration is under-determined: the ACS block "
                f"{tuple(acs.shape[:3])} is too small for kernel_size "
                f"{tuple(kernel_size)} at acceleration {tuple(af)}. Use a "
                f"smaller kernel or acquire more ACS lines."
            ) from exc
        raise
    # Back to [nx, ny, nz, ncoils].
    return filled.permute(3, 1, 2, 0).contiguous().cpu()


# =====================================================================
# Partial Fourier: POCS
# =====================================================================

def get_partial_fourier_mask(kspace: torch.Tensor, dim_pf: int) -> torch.Tensor:
    """1D mask of acquired positions along the partial-Fourier axis.

    Assumes that unacquired positions are zero across all other axes/coils.
    """
    reduction_dims = tuple(a for a in range(kspace.ndim) if a != dim_pf)
    return torch.any(torch.abs(kspace) > 0, dim=reduction_dims)


def pocs_partial_fourier(kspace: torch.Tensor,
                         dim_enc: Sequence[int] = ENCODING_DIMS,
                         dim_pf: int = 0,
                         number_of_iterations: int = 10,
                         gaussian_std: float = 10.0,
                         center_index: Optional[int] = None,
                         device: str = "cpu") -> torch.Tensor:
    """Partial-Fourier reconstruction by POCS (projection onto convex sets).

    A real-valued object has a Hermitian-symmetric k-space,
    ``S(-k) = conj(S(k))``, so roughly half of k-space is redundant. MR
    images are not real, but their phase is usually smooth. POCS therefore

    1. estimates a low-resolution phase map ``phi`` from the symmetric,
       fully acquired centre of k-space (Gaussian-weighted to avoid ringing);
    2. alternates between two projections:

       * phase constraint: in image space, keep ``|x|`` and replace the phase
         by ``phi``  (``x <- |x| * exp(i * phi)``);
       * data consistency: in k-space, put the acquired samples back.

    Only the partial-Fourier axis is iterated; the other encoding axes are
    transformed once to image space and stay there during the loop (a
    "hybrid" space), which is exact because the sampling mask only varies
    along ``dim_pf``. The estimated samples fill the missing edge; acquired
    samples are returned unchanged (checked at the end).

    Parameters
    ----------
    kspace:
        Complex k-space ``[nx, ny, nz, ncoils]``.
    dim_enc:
        Fourier-encoded axes (the coil axis must not be included).
    dim_pf:
        Partial-Fourier axis (0 = readout / asymmetric echo, 1 = phase,
        2 = partition).
    number_of_iterations:
        Number of POCS iterations (>= 1).
    gaussian_std:
        Standard deviation, in samples, of the Gaussian window applied to the
        symmetric centre for phase estimation.
    center_index:
        Index of k = 0 along ``dim_pf`` in the *current* array
        (default ``n // 2``).
    device:
        Device used for the computation.

    Assumptions: one-sided partial Fourier along a single axis, no internal
    gaps along that axis, a symmetric acquired centre, and more than 50 % of
    the positions acquired.
    """
    if not isinstance(kspace, torch.Tensor):
        kspace = torch.as_tensor(kspace)
    if not torch.is_complex(kspace):
        raise TypeError("POCS requires complex-valued k-space.")
    if kspace.ndim != 4:
        raise ValueError(f"Expected 4D k-space, got {tuple(kspace.shape)}")
    if number_of_iterations < 1:
        raise ValueError("number_of_iterations must be >= 1.")
    if gaussian_std <= 0:
        raise ValueError("gaussian_std must be positive.")

    original_device = kspace.device
    original_dtype = kspace.dtype
    kspace = kspace.to(device)

    ndim = kspace.ndim
    dim_pf = dim_pf % ndim
    dim_enc = tuple(sorted(set(int(d) % ndim for d in dim_enc)))
    if dim_pf not in dim_enc:
        raise ValueError("dim_pf must belong to dim_enc.")
    if ndim - 1 in dim_enc:
        raise ValueError("The coil dimension must not be Fourier transformed.")
    nonpf_enc_dims = tuple(d for d in dim_enc if d != dim_pf)

    n_pf = kspace.shape[dim_pf]
    if center_index is None:
        center_index = n_pf // 2
    center_index = int(center_index)
    if not 0 <= center_index < n_pf:
        raise ValueError("Invalid partial-Fourier centre index.")

    # 1. Acquisition mask along the PF axis ------------------------------
    mask_original = get_partial_fourier_mask(kspace, dim_pf)
    acquired_indices = torch.where(mask_original)[0]
    if acquired_indices.numel() == 0:
        raise RuntimeError("No acquired partial-Fourier positions detected.")
    first_acquired = int(acquired_indices[0].item())
    last_acquired = int(acquired_indices[-1].item())
    acquired_fraction = acquired_indices.numel() / n_pf
    if acquired_fraction >= 1.0:
        raise RuntimeError("No missing partial-Fourier positions detected.")
    if acquired_fraction <= 0.5:
        raise RuntimeError("POCS requires more than 50% of the PF positions.")
    if not mask_original[center_index]:
        raise RuntimeError("PF centre is not acquired; phase estimation is unsafe.")

    # 2. Contiguous one-sided acquisition support -------------------------
    support_mask = torch.zeros(n_pf, dtype=torch.bool, device=kspace.device)
    support_mask[first_acquired:last_acquired + 1] = True
    if (~mask_original[first_acquired:last_acquired + 1]).any():
        raise RuntimeError(
            "Internal missing PF positions detected. This POCS implementation "
            "assumes a contiguous one-sided partial-Fourier acquisition."
        )
    missing_left = first_acquired > 0
    missing_right = last_acquired < n_pf - 1
    if missing_left and missing_right:
        raise RuntimeError(
            "Missing samples on both sides; only one-sided partial Fourier is supported."
        )

    # 3. Symmetric central region: k and its mirror -k both acquired -----
    indices = torch.arange(n_pf, device=kspace.device)
    reflected = 2 * center_index - indices
    valid = (reflected >= 0) & (reflected < n_pf)
    symmetric_mask = torch.zeros(n_pf, dtype=torch.bool, device=kspace.device)
    symmetric_mask[indices[valid]] = support_mask[indices[valid]] & support_mask[reflected[valid]]
    if symmetric_mask.sum() < 3:
        raise RuntimeError("Insufficient symmetric central region.")

    # 4. Gaussian-weighted symmetric k-space (low-resolution phase) ------
    coordinates = torch.arange(n_pf, dtype=torch.float32, device=kspace.device)
    gaussian = torch.exp(-0.5 * ((coordinates - center_index) / gaussian_std) ** 2)
    gaussian = gaussian * symmetric_mask.float()
    view_shape = [1] * ndim
    view_shape[dim_pf] = n_pf
    kspace_symmetric = kspace * gaussian.reshape(view_shape)

    # 5. Phase estimate, in full image space (all encoding axes) ----------
    phase_estimate = centered_ifft(kspace_symmetric, dims=dim_enc)
    phase_magnitude = torch.abs(phase_estimate)
    maximum = phase_magnitude.max()
    if maximum <= 0:
        raise RuntimeError("Phase estimation failed: symmetric k-space is zero.")
    valid_phase = phase_magnitude > maximum * 1e-6
    phase_factor = torch.ones_like(phase_estimate)  # exp(i * phi)
    phase_factor[valid_phase] = phase_estimate[valid_phase] / phase_magnitude[valid_phase]

    # 6. Hybrid space: image along non-PF axes, k-space along the PF axis -
    kspace_hybrid = centered_ifft(kspace, dims=nonpf_enc_dims)
    kspace_hybrid_original = kspace_hybrid.clone()
    mask_view = mask_original.reshape(view_shape)

    # 7. POCS iterations --------------------------------------------------
    for _ in range(number_of_iterations):
        image = centered_ifft(kspace_hybrid, dims=(dim_pf,))      # full image
        image = torch.abs(image) * phase_factor                     # phase constraint
        kspace_hybrid = centered_fft(image, dims=(dim_pf,))         # back to hybrid
        kspace_hybrid = torch.where(mask_view, kspace_hybrid_original,
                                    kspace_hybrid)                  # data consistency

    # 8. Back to fully encoded k-space -----------------------------------
    kspace_reconstructed = centered_fft(kspace_hybrid, dims=nonpf_enc_dims)

    # 9. Acquired samples must be unchanged (up to FFT round-off) --------
    mask_full = mask_view.expand_as(kspace)
    consistency_error = torch.max(torch.abs(kspace_reconstructed[mask_full] - kspace[mask_full]))
    tolerance = 1e-4 * max(float(torch.abs(kspace[mask_full]).max().item()), 1e-12)
    if float(consistency_error.item()) > tolerance:
        raise RuntimeError(
            "POCS data-consistency check failed: "
            f"maximum difference = {consistency_error.item():.6e}"
        )
    return kspace_reconstructed.to(device=original_device, dtype=original_dtype)


# =====================================================================
# SimpleITK conversion (for 3D Slicer)
# =====================================================================

def center_origin(image: sitk.Image) -> sitk.Image:
    """Set the origin so that physical (0, 0, 0) is at the volume centre.

    SimpleITK's origin is the centre of the first voxel, hence the
    ``- 0.5 * spacing`` term.
    """
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    size = np.array(image.GetSize(), dtype=float)
    half_size = 0.5 * size * spacing
    image.SetOrigin(tuple(-direction @ (half_size - 0.5 * spacing)))
    return image


def make_sitk_image(volume: torch.Tensor,
                    spacing: Sequence[float] = (1.0, 1.0, 1.0),
                    origin: Sequence[float] = (0.0, 0.0, 0.0),
                    direction: Sequence[float] = IDENTITY_DIRECTION,
                    center_volume: bool = False) -> sitk.Image:
    """Convert a real ``[z, y, x]`` tensor to a float32 SimpleITK image.

    SimpleITK reads numpy arrays as ``[z, y, x]``, so callers permute
    ``[x, y, z]`` volumes with ``volume.permute(2, 1, 0)`` first.
    ``center_volume=True`` ignores ``origin`` and centres the volume on
    (0, 0, 0) (see :func:`center_origin`).
    """
    data = volume.detach().cpu().numpy().astype(np.float32)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(tuple(float(s) for s in spacing))
    image.SetDirection(tuple(float(d) for d in direction))
    if center_volume:
        center_origin(image)
    else:
        image.SetOrigin(tuple(float(o) for o in origin))
    return image


def kspace_to_display(kspace: torch.Tensor) -> torch.Tensor:
    """Log-compressed RSS magnitude of k-space, normalised to [0, 1].

    ``[nx, ny, nz, ncoils]`` -> ``[nx, ny, nz]``. ``log1p`` compresses the
    large dynamic range between the k-space centre and its periphery.
    """
    if kspace.dim() == 3:
        kspace = kspace.unsqueeze(-1)
    display = torch.log1p(rss(kspace, dim=-1))
    display = display - display.min()
    maximum = display.max()
    if maximum > 0:
        display = display / maximum
    return display


def kspace_to_sitk(kspace: torch.Tensor, center_volume: bool = True) -> sitk.Image:
    """K-space ``[nx, ny, nz, ncoils]`` -> SimpleITK display volume (unit spacing)."""
    volume = kspace_to_display(kspace).permute(2, 1, 0).float()
    return make_sitk_image(volume, center_volume=center_volume)
