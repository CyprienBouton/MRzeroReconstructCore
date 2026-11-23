"""Low-level reconstruction of MRzero simulated signals.

:class:`RecoMRzero` grids the simulated ADC samples of an MRzero sequence onto
a Cartesian k-space and provides the reconstruction back-ends used by
:class:`MRzeroReconstructCoreLib.MRzeroReconstructLogic`:

* centred inverse FFT,
* GRAPPA (vendored ``ggrappa``),
* wavelet-regularized compressed sensing after discarding the samples acquired
  with an irregular recovery time ("Zero-filled" method).

K-space arrays follow the conventions documented in :mod:`.reco_utils`
(``[nx, ny, nz, ncoils]``, k = 0 at index ``n // 2`` once centred).
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from . import reco_utils
from .compressed_sensing import reconstruct_cartesian_cs


def get_IR(seq0) -> Tuple[torch.Tensor, list]:
    """Start times and indices of the inversion pulses of a sequence.

    Inversion pulses are the repetitions whose flip angle equals the largest
    flip angle of the sequence.

    Args:
        seq0 (MRzeroCore.Sequence): MRzero sequence.

    Returns:
        (times_IR, idx_IRs): start time [s] of each inversion repetition and
        its index in ``seq0``.
    """
    angle_max = max(float(r.pulse.angle) for r in seq0)
    idx_IRs = [i for i, r in enumerate(seq0) if float(r.pulse.angle) == angle_max]
    times_before = _repetition_start_times(seq0)
    return times_before[idx_IRs], idx_IRs


def _repetition_start_times(seq0) -> torch.Tensor:
    """Start time [s] of every repetition (cumulative sum of event times)."""
    durations = torch.stack([torch.as_tensor(r.event_time).sum() for r in seq0])
    times_after = torch.cumsum(durations, dim=0)
    return torch.cat([torch.zeros(1, dtype=times_after.dtype), times_after[:-1]])


def _adc_count(rep) -> int:
    """Number of measured samples of a repetition (``adc_usage > 0``)."""
    return int((rep.adc_usage > 0).sum())


def extract_indices_from_kspace_coords(kspace_coords: torch.Tensor, dk: float) -> torch.Tensor:
    """Integer grid index of every k-space coordinate along one axis.

    The sorted unique coordinates are converted to indices by rounding every
    gap between consecutive coordinates to a multiple of ``dk`` and summing
    the rounded gaps. Rounding each gap individually avoids the accumulation
    of floating-point error of ``round((k - k_min) / dk)`` and keeps
    undersampling gaps (e.g. GRAPPA) on the grid.

    Args:
        kspace_coords: 1D tensor of k-space coordinates.
        dk: k-space step along this axis.

    Returns:
        1D integer tensor with the same shape as ``kspace_coords``.
    """
    unique_coords, inverse = torch.unique(kspace_coords, sorted=True, return_inverse=True)
    gaps = torch.round(torch.diff(unique_coords) / dk).to(torch.long)
    gaps = torch.cat([torch.zeros(1, dtype=torch.long, device=kspace_coords.device), gaps])
    return torch.cumsum(gaps, dim=0)[inverse]


def _step_near_zero(unique_values: torch.Tensor, default: float) -> torch.Tensor:
    """k-space step measured next to k = 0 (assumed fully sampled there)."""
    if unique_values.numel() < 2:
        return torch.as_tensor(default)
    steps = unique_values.diff()
    index = min(int(unique_values.abs().argmin()), steps.numel() - 1)
    return steps[index]


class RecoMRzero:
    """Cartesian reconstruction of an MRzero simulated signal.

    Typical use::

        reco = RecoMRzero(seq)
        reco.remove_navigator()                      # optional
        raw, kx, ky, kz = reco.cartesian_kspace_grid(signal)
        kspace, center = reco.center_kspace(raw)
        image = reco.fft_reconstruct(kspace)

    Args:
        seq0: ``MRzeroCore.Sequence`` (loaded from ``.seq`` or ``.pkl``).
        dx, dy, dz: k-space steps [1/m]. ``None`` measures them next to k = 0
            (which must therefore be fully sampled).
    """

    #: Assumed slice thickness when the sequence has a single partition.
    DEFAULT_SINGLE_SLICE_THICKNESS = 5e-3  # [m]

    def __init__(self, seq0, dx=None, dy=None, dz=None):
        self.seq0 = seq0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        # Set by cartesian_kspace_grid():
        self.ix = self.iy = self.iz = None
        self.nx = self.ny = self.nz = None
        self.ncoils = None
        self.kspace_center = None
        # Set by remove_navigator(): signal samples to keep.
        self.signal_keep_mask = None

    # ================================================================
    # SEQUENCE PREPARATION
    # ================================================================

    def remove_navigator(self) -> bool:
        """Discard the navigator blocks of inversion-recovery sequences.

        The navigator is detected when the inversion pulses (largest flip
        angle) come in pairs two repetitions apart: for every first pulse
        ``i`` of a pair, the ADC of repetition ``i + 2`` is disabled and the
        flip angles of repetitions ``i + 1`` and ``i + 2`` are set to zero, so
        that :meth:`cartesian_kspace_grid` and the inversion-time analysis
        ignore the navigator.

        The sequence is modified in place. The signal samples that belonged
        to the disabled ADCs are recorded in :attr:`signal_keep_mask` and
        dropped by :meth:`cartesian_kspace_grid` if the signal still
        contains them.

        Returns:
            True if a navigator pattern was found and removed.
        """
        _, idx_IRs = get_IR(self.seq0)
        unique_gaps = np.unique(np.diff(idx_IRs))
        if len(unique_gaps) <= 1 or unique_gaps[0] != 2:
            return False

        adc_counts = [_adc_count(r) for r in self.seq0]
        removed = set()
        for i in idx_IRs[::2]:
            if i + 2 >= len(self.seq0):
                break
            navigator = self.seq0[i + 2]
            # NOTE: the original code zeroed `seq0[i + 2].adc_usage` twice;
            # whether `seq0[i + 1].adc_usage` was intended is unverified.
            navigator.adc_usage = torch.zeros_like(navigator.adc_usage)
            self.seq0[i + 1].pulse.angle = torch.zeros_like(torch.as_tensor(self.seq0[i + 1].pulse.angle))
            navigator.pulse.angle = torch.zeros_like(torch.as_tensor(navigator.pulse.angle))
            removed.add(i + 2)

        self.signal_keep_mask = torch.cat([
            torch.full((count,), index not in removed, dtype=torch.bool)
            for index, count in enumerate(adc_counts)
        ])
        return True

    # ================================================================
    # GRIDDING
    # ================================================================

    def cartesian_kspace_grid(self, signal: torch.Tensor, precision: float = 5e-2,
                              dtype=torch.complex64, device="cpu"):
        """Grid the simulated signal onto a Cartesian k-space.

        The k-space position of every ADC sample comes from
        ``seq0.get_kspace()`` [1/m]. Coordinates are rounded to ``precision``
        and converted to grid indices with
        :func:`extract_indices_from_kspace_coords`. Samples falling on the same
        grid point are summed. Positions that were not acquired stay zero.

        Args:
            signal: ``[n_samples, ncoils]`` (or ``[n_samples]``) complex signal.
            precision: rounding step of the k-space coordinates [1/m].

        Returns:
            grid: ``[nx, ny, nz, ncoils]`` k-space (not centred).
            kx_vals, ky_vals, kz_vals: sorted unique rounded coordinates.

        Side effects:
            Sets ``ix, iy, iz`` (grid index of every sample), ``nx, ny, nz,
            ncoils``, the steps ``dx, dy, dz`` and :attr:`kspace_center`
            (grid index of the sample closest to k = 0).
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(-1)

        kspace_coords = self.seq0.get_kspace()
        n_samples = kspace_coords.shape[0]
        if signal.shape[0] != n_samples:
            keep = self.signal_keep_mask
            if keep is not None and signal.shape[0] == keep.numel() and int(keep.sum()) == n_samples:
                signal = signal[keep]
            else:
                raise ValueError(
                    f"The signal has {signal.shape[0]} samples but the sequence "
                    f"has {n_samples} ADC samples."
                )

        kx, ky, kz = kspace_coords[:, :3].T
        kx_rounded = (kx / precision).round() * precision
        ky_rounded = (ky / precision).round() * precision
        kz_rounded = (kz / precision).round() * precision
        kx_unique = kx_rounded.unique(sorted=True)
        ky_unique = ky_rounded.unique(sorted=True)
        kz_unique = kz_rounded.unique(sorted=True)

        # Steps measured next to k = 0, which must be fully sampled.
        if self.dx is None:
            self.dx = _step_near_zero(kx_unique, 1.0)
        if self.dy is None:
            self.dy = _step_near_zero(ky_unique, 1.0)
        if self.dz is None:
            # A single partition (2D) gets a nominal slice thickness:
            # spacing = 1 / (dz * nz) = DEFAULT_SINGLE_SLICE_THICKNESS.
            self.dz = _step_near_zero(kz_unique, 1.0 / self.DEFAULT_SINGLE_SLICE_THICKNESS)

        self.ix = extract_indices_from_kspace_coords(kx_rounded, self.dx)
        self.iy = extract_indices_from_kspace_coords(ky_rounded, self.dy)
        self.iz = extract_indices_from_kspace_coords(kz_rounded, self.dz)

        self.nx = int(self.ix.max()) + 1
        self.ny = int(self.iy.max()) + 1
        self.nz = int(self.iz.max()) + 1
        self.ncoils = signal.shape[1]

        # Grid index of k = 0 (closest sample) along each axis.
        self.kspace_center = (
            int(self.ix[kx_rounded.abs().argmin()]),
            int(self.iy[ky_rounded.abs().argmin()]),
            int(self.iz[kz_rounded.abs().argmin()]),
        )

        grid = torch.zeros((self.nx, self.ny, self.nz, self.ncoils), dtype=dtype, device=device)
        grid.index_put_((self.ix, self.iy, self.iz), signal.to(dtype=dtype, device=device),
                        accumulate=True)
        return grid, kx_unique, ky_unique, kz_unique

    def center_kspace(self, kspace: torch.Tensor):
        """Zero-pad the gridded k-space so that k = 0 is at index ``n // 2``.

        Returns ``(centered_kspace, new_center)``; see
        :func:`.reco_utils.center_kspace`. No sample is discarded.
        """
        if self.kspace_center is None:
            raise RuntimeError("Call cartesian_kspace_grid() first.")
        return reco_utils.center_kspace(kspace, self.kspace_center)

    # ================================================================
    # RECONSTRUCTION BACK-ENDS
    # ================================================================

    def nufft_reconstruct(self, signal: torch.Tensor) -> torch.Tensor:
        """Density-compensated adjoint NUFFT on the inferred Cartesian image grid.

        Uses exact sequence coordinates, with field of view inferred from the
        Cartesian k-space steps. Call cartesian_kspace_grid first.
        """
        from torchkbnufft import KbNufftAdjoint, calc_density_compensation_function

        if self.ix is None:
            raise RuntimeError("Call cartesian_kspace_grid() first.")
        if signal.ndim == 1:
            signal = signal[:, None]
        coords = self.seq0.get_kspace()[:, :3].to(signal.device)
        if len(signal) != len(coords) and self.signal_keep_mask is not None:
            signal = signal[self.signal_keep_mask.to(signal.device)]
        if len(signal) != len(coords):
            raise ValueError("Signal and trajectory sample counts differ.")
        shape = self.center_kspace(torch.zeros(
            (self.nx, self.ny, self.nz), device=signal.device))[0].shape
        axes = [axis for axis, size in enumerate(shape) if size > 1]
        sizes = tuple(shape[axis] for axis in axes)
        steps = (self.dx, self.dy, self.dz)
        omega = torch.stack([2 * torch.pi * coords[:, axis] / float(steps[axis]) / shape[axis]
                             for axis in axes]).to(torch.float32)
        adjoint = KbNufftAdjoint(im_size=sizes).to(signal.device)
        weights = calc_density_compensation_function(omega, im_size=sizes)
        samples = signal.T[None].to(torch.complex64)
        image = adjoint(samples * weights, omega, norm="ortho")[0]
        return image.movedim(0, -1).reshape(*shape, signal.shape[-1])

    def fft_reconstruct(self, kspace: torch.Tensor) -> torch.Tensor:
        """Unitary centred inverse FFT over the three encoding axes.

        Args:
            kspace: centred k-space ``[nx, ny, nz, ncoils]``.

        Returns:
            Complex coil images ``[nx, ny, nz, ncoils]``.
        """
        return reco_utils.centered_ifft(kspace, dims=(0, 1, 2))

    def get_acs_from_kspace(self, kspace: torch.Tensor,
                            center: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Largest fully sampled block around the centre of a centred k-space."""
        return reco_utils.get_acs_from_kspace(kspace, center)

    def grappa_reconstruct(self, kspace: torch.Tensor, acs: torch.Tensor,
                           af: Optional[Sequence[int]] = None,
                           kernel_size: Sequence[int] = (4, 4, 5),
                           reg: float = 1e-4, batch_size: int = 1,
                           cuda: Optional[bool] = None, quiet: bool = True) -> torch.Tensor:
        """GRAPPA k-space filling; returns k-space ``[nx, ny, nz, ncoils]``.

        See :func:`.reco_utils.grappa_reconstruct` for the maths and the
        parameters. ``af=None`` estimates the acceleration from the sampling
        pattern; ``cuda=None`` uses the GPU when available.
        """
        return reco_utils.grappa_reconstruct(kspace, acs, af=af, kernel_size=kernel_size,
                                             reg=reg, batch_size=batch_size, cuda=cuda,
                                             quiet=quiet)

    def zero_filled_cs(self, kspace: torch.Tensor, tolerance: float = 0.15,
                       regularization: float = 0.1, reference_scan: str = "mode",
                       calib_kernel: str = "6:6:2") -> torch.Tensor:
        """Compressed sensing after discarding irregular-recovery samples.

        In inversion-recovery sequences with a variable delay between shots,
        each k-space line inherits the time since the previous inversion of
        its shot (:meth:`get_inter_scan_duration`). Lines whose duration
        differs from the reference duration by more than
        ``tolerance * reference`` are excluded from the data fidelity term.
        Coil sensitivities come from central k-space before that exclusion.

        Args:
            kspace: *raw* (not centred) gridded k-space from
                :meth:`cartesian_kspace_grid`.
            tolerance: relative tolerance on the recovery duration.
            regularization: L1-wavelet weight ``lambda``.
            reference_scan: ``"mode"`` (most frequent duration) or ``"min"``.
            calib_kernel: Colon-separated widths of the central calibration window (x:y:z).

        Returns:
            Complex coil-combined image ``[nx, ny, nz]`` on the centred grid.
        """
        inter_scan_matrix = self.get_inter_scan_duration()
        durations = torch.round(inter_scan_matrix.flatten() * 10) / 10
        durations = durations[~torch.isnan(durations)]
        if durations.numel() == 0:
            raise ValueError(
                "No inversion pulse found: the 'Zero-filled' method needs an "
                "inversion-recovery sequence."
            )
        if reference_scan == "mode":
            reference, _ = torch.mode(durations)
        elif reference_scan == "min":
            reference = durations.min()
        else:
            raise ValueError("reference_scan must be 'mode' or 'min'")

        # [ny, nz] mask of lines acquired with an irregular recovery time.
        # NaN durations (lines acquired before the first inversion) compare
        # False and are therefore kept.
        bad_mask = torch.abs(inter_scan_matrix - reference) > tolerance * reference
        acquired = torch.zeros(kspace.shape[:3], dtype=torch.bool, device=kspace.device)
        acquired[self.ix, self.iy, self.iz] = True
        retained = acquired & ~bad_mask.to(kspace.device).unsqueeze(0)

        centered, _ = self.center_kspace(kspace)
        acquired, _ = self.center_kspace(acquired)
        retained, _ = self.center_kspace(retained)
        return reconstruct_cartesian_cs(centered, acquired, retained, regularization,
                                        calib_kernel)

    def get_inter_scan_duration(self, eps: float = 0.2, dtype=torch.float,
                                device="cpu") -> torch.Tensor:
        """Duration between consecutive inversions, per k-space line.

        Inversion pulses are the repetitions with a flip angle within ``eps``
        rad of pi. Each ADC sample is assigned to the shot started by the
        last inversion before it; the line ``(iy, iz)`` of the sample gets the
        time elapsed between that inversion and the previous one.

        Returns:
            ``[ny, nz]`` tensor [s]; NaN for lines without a preceding pair
            of inversions.
        """
        if self.iy is None:
            raise RuntimeError("Call cartesian_kspace_grid() first.")
        inter_scan_matrix = torch.full((self.ny, self.nz), torch.nan, dtype=dtype, device=device)
        times_before = _repetition_start_times(self.seq0)
        idx_IR = [i for i, r in enumerate(self.seq0)
                  if abs(float(r.pulse.angle) - np.pi) < eps]
        times_IR = times_before[idx_IR]
        durations = torch.diff(times_IR, prepend=torch.tensor([torch.nan], dtype=times_IR.dtype))
        times_IR_next = torch.cat([times_IR[1:], torch.tensor([torch.inf], dtype=times_IR.dtype)])
        adc_counts = [_adc_count(r) for r in self.seq0]
        for time_IR, time_IR_next, duration in zip(times_IR, times_IR_next, durations):
            in_shot = (times_before > time_IR) & (times_before < time_IR_next)
            mask_signal = torch.cat([
                torch.full((count,), bool(flag), dtype=torch.bool)
                for flag, count in zip(in_shot, adc_counts)
            ])
            inter_scan_matrix[self.iy[mask_signal], self.iz[mask_signal]] = duration.to(dtype)
        return inter_scan_matrix

    # ================================================================
    # GEOMETRY
    # ================================================================

    def get_spacing(self, shape: Optional[Sequence[int]] = None) -> Tuple[float, float, float]:
        """Voxel size [mm] of an image reconstructed on a grid of ``shape``.

        The field of view along an axis is ``1 / dk``, so a grid of ``n``
        points has a voxel size of ``1 / (dk * n)`` m. ``shape`` defaults to
        the gridded size ``(nx, ny, nz)``; pass the centred (padded) shape
        when the k-space was zero-padded.
        """
        if shape is None:
            shape = (self.nx, self.ny, self.nz)
        steps = (self.dx, self.dy, self.dz)
        return tuple(float(1.0 / (float(dk) * int(n))) * 1e3 for dk, n in zip(steps, shape[:3]))
