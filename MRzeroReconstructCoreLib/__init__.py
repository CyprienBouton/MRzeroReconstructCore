"""MRzero reconstruction core.

Reconstructs MRzero simulated MRI signals (FFT, GRAPPA, wavelet-regularized compressed
sensing) into SimpleITK volumes. Used by the 3D Slicer extension
``MRzeroReconstruct``.

Public API::

    from MRzeroReconstructCoreLib import MRzeroReconstructLogic
    image, kspace, raw_kspace = MRzeroReconstructLogic().runReconstruction(
        "signal.npy", "sequence.seq", "FFT")
"""

__version__ = "0.3.0"

from .MRzeroReconstructCore import METHODS, MRzeroReconstructLogic, load_sequence
from .RecoMRzero import RecoMRzero

__all__ = ["MRzeroReconstructLogic", "RecoMRzero", "METHODS", "load_sequence", "__version__"]
