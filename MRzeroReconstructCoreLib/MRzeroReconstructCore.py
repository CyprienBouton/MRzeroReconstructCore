"""High-level reconstruction entry point for MRzero simulated signals.

:class:`MRzeroReconstructLogic` is the API used by the 3D Slicer extension
``MRzeroReconstruct``. It loads a sequence (``.seq`` or pickled
``MRzeroCore.Sequence``) and a simulated signal (``.npy``), reconstructs it
and returns SimpleITK images ready to be loaded in Slicer.
"""

import os
import pickle
from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk
import torch

from . import reco_utils
from .RecoMRzero import RecoMRzero

#: Reconstruction methods accepted by ``runReconstruction`` (case-insensitive).
METHODS = ("FFT", "GRAPPA", "NUFFT", "Zero-filled")

#: Orientation of reconstructed MRzero images (x axis flipped for Slicer's RAS display).
IMAGE_DIRECTION = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def load_sequence(sequencePath: str):
    """Load an MRzero sequence from a Pulseq ``.seq`` file or a pickled ``.pkl``.

    ``.pkl`` files are unpickled: only load files you trust.
    """
    sequencePath = os.path.normpath(os.fspath(sequencePath))
    if sequencePath.endswith(".seq"):
        import MRzeroCore as mr0
        return mr0.Sequence.import_file(sequencePath)
    if sequencePath.endswith(".pkl"):
        import MRzeroCore  # noqa: F401  (classes needed to unpickle the sequence)
        with open(sequencePath, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported sequence file format: {sequencePath} (expected .seq or .pkl)")


class MRzeroReconstructLogic:
    """Reconstruct MRzero simulated signals into SimpleITK volumes."""

    # ================================================================
    # HELPERS (kept as static methods for backward compatibility)
    # ================================================================

    @staticmethod
    def add_gaussian_noise(signal: np.ndarray, signal_ref: np.ndarray,
                           noise_factor: float) -> np.ndarray:
        """Add complex white Gaussian noise to a simulated signal.

        The standard deviations of the real and imaginary noise are
        ``noise_factor * max(Re(signal_ref))`` and
        ``noise_factor * max(Im(signal_ref))`` respectively.
        """
        noise_std_real = signal_ref.real.max() * noise_factor
        noise_std_imag = signal_ref.imag.max() * noise_factor
        real = signal.real + np.random.normal(0, noise_std_real, signal.real.shape)
        imag = signal.imag + np.random.normal(0, noise_std_imag, signal.imag.shape)
        return real + 1j * imag

    @staticmethod
    def kspace_to_sitk(kspace: torch.Tensor) -> sitk.Image:
        """Multi-coil k-space -> log-compressed RSS display volume (centred, unit spacing)."""
        return reco_utils.kspace_to_sitk(kspace, center_volume=True)

    @staticmethod
    def center_origin(sitk_img: sitk.Image) -> sitk.Image:
        """Put physical (0, 0, 0) at the centre of the volume."""
        return reco_utils.center_origin(sitk_img)

    @staticmethod
    def make_sitk_image(volume: torch.Tensor,
                        spacing: Sequence[float] = (1.0, 1.0, 1.0),
                        origin: Sequence[float] = (0.0, 0.0, 0.0),
                        direction: Sequence[float] = reco_utils.IDENTITY_DIRECTION,
                        center_volume: bool = False) -> sitk.Image:
        """Real ``[z, y, x]`` tensor -> float32 SimpleITK image."""
        return reco_utils.make_sitk_image(volume, spacing=spacing, origin=origin,
                                          direction=direction, center_volume=center_volume)

    # ================================================================
    # MAIN RECONSTRUCTION
    # ================================================================

    def runReconstruction(
        self,
        signalPath: str,
        sequencePath: str,
        method: str = "FFT",
        kernelSize: Sequence[int] = (4, 1, 5),
        reg: float = 1e-4,
        tol: float = 0.1,
        reg_cs: float = 0.1,
        reference_scan: str = "mode",
        signalPathRef: Optional[str] = None,
        gaussian_noise_factor: float = 0.0,
        doRemoveNavigator: bool = True,
        af: Optional[Sequence[int]] = None,
    ):
        """Reconstruct one simulated signal.

        Args:
            signalPath: ``.npy`` simulated signal ``[n_samples, ncoils]``.
            sequencePath: ``.seq`` or ``.pkl`` MRzero sequence used for the
                simulation.
            method: ``"FFT"``, ``"GRAPPA"`` or ``"Zero-filled"``
                (case-insensitive).
            kernelSize: GRAPPA kernel ``(ky, kz, kx)``.
            reg: GRAPPA Tikhonov regularisation.
            tol: Zero-filled: relative tolerance on the recovery duration.
            reg_cs: Zero-filled: L1-wavelet regularisation.
            reference_scan: Zero-filled: reference duration, ``"mode"`` or
                ``"min"``.
            signalPathRef: optional reference signal whose maximum sets the
                noise level; defaults to the signal itself.
            gaussian_noise_factor: noise standard deviation relative to the
                maximum of the reference signal (0 = no noise).
            doRemoveNavigator: discard navigator blocks
                (:meth:`RecoMRzero.remove_navigator`).
            af: GRAPPA acceleration ``(af_y, af_z)``; ``None`` = estimated.

        Returns:
            ``(image_sitk, reconstruction_kspace_sitk, raw_kspace_sitk)``:
            the magnitude image (RSS over coils, physical spacing in mm,
            centred on the origin), the k-space corresponding to that image
            and the gridded raw k-space, both as log-magnitude display
            volumes.
        """
        method_key = method.strip().upper()
        if method_key not in {m.upper() for m in METHODS}:
            raise ValueError(
                f"Unknown reconstruction method: {method}. "
                f"Available methods: {', '.join(METHODS)}."
            )

        reco = RecoMRzero(load_sequence(sequencePath))

        signal_np = np.load(signalPath)
        if gaussian_noise_factor > 0.0:
            signal_ref_np = np.load(signalPathRef) if signalPathRef is not None else signal_np
            signal_np = self.add_gaussian_noise(signal_np, signal_ref_np, gaussian_noise_factor)
        signal = torch.from_numpy(np.asarray(signal_np))

        # ------------------------------------------------------------
        # Raw Cartesian k-space
        # ------------------------------------------------------------
        if doRemoveNavigator:
            reco.remove_navigator()
        raw_kspace, _, _, _ = reco.cartesian_kspace_grid(signal)
        raw_kspace_sitk = self.kspace_to_sitk(raw_kspace)

        # ------------------------------------------------------------
        # Reconstruction
        # ------------------------------------------------------------
        if method_key == "FFT":
            reconstruction_kspace, _ = reco.center_kspace(raw_kspace)
            output = reco.fft_reconstruct(reconstruction_kspace)

        elif method_key == "GRAPPA":
            kspace, center = reco.center_kspace(raw_kspace)
            acs = reco.get_acs_from_kspace(kspace, center)
            reconstruction_kspace = reco.grappa_reconstruct(
                kspace, acs, af=af, kernel_size=kernelSize, reg=reg
            )
            output = reco.fft_reconstruct(reconstruction_kspace)

        elif method_key == "NUFFT":
            output = reco.nufft_reconstruct(signal)
            reconstruction_kspace = reco_utils.centered_fft(output)

        else:  # "ZERO-FILLED"
            output = reco.zero_filled_cs(
                raw_kspace,
                tolerance=tol,
                regularization=reg_cs,
                reference_scan=reference_scan,
            )
            # K-space consistent with the coil-combined CS image.
            reconstruction_kspace = reco_utils.centered_fft(output, dims=(0, 1, 2))

        # ------------------------------------------------------------
        # Coil combination and conversion
        # ------------------------------------------------------------
        magnitude = reco_utils.rss(output, dim=-1) if output.dim() == 4 else output.abs()
        image_sitk = self.make_sitk_image(
            magnitude.permute(2, 1, 0),
            spacing=reco.get_spacing(magnitude.shape),
            direction=IMAGE_DIRECTION,
            center_volume=True,
        )
        reconstruction_kspace_sitk = self.kspace_to_sitk(reconstruction_kspace)

        return image_sitk, reconstruction_kspace_sitk, raw_kspace_sitk

    # ================================================================
    # MULTIPLE-FILE RECONSTRUCTION
    # ================================================================

    def runMultipleReconstructions(self, signalPaths, sequencePath, method="FFT", **kwargs):
        """Run :meth:`runReconstruction` on several signals with one sequence.

        Extra keyword arguments are forwarded to :meth:`runReconstruction`.
        Errors are caught per file.

        Returns:
            list of ``{"filename", "result", "error"}`` dicts, where ``result``
            is the tuple returned by :meth:`runReconstruction` (or None) and
            ``error`` the error message (or None).
        """
        results = []
        for signalPath in signalPaths:
            try:
                result = self.runReconstruction(signalPath, sequencePath, method, **kwargs)
                results.append({"filename": signalPath, "result": result, "error": None})
            except Exception as exc:  # keep going with the other files
                results.append({"filename": signalPath, "result": None, "error": str(exc)})
        return results
