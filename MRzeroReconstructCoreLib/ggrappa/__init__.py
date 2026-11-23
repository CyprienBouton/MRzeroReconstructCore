# Vendored from gGRAPPA (upstream: https://github.com/mbertrait/ggrappa) via the fork
# https://github.com/CyprienBouton/ggrappa, commit 39cfab0 (2025-09-19).
# BSD 3-Clause License, Copyright (c) 2024, gGRAPPA Collaborators; see LICENSE in
# this directory. The code below is unmodified apart from this header.

"""gGRAPPA.

gGRAPPA provides an easy to use generalized GRAPPA reconstruction tool for kspace data.
"""
import torch

from dataclasses import dataclass
from typing import List


@dataclass
class GRAPPAReconSpec:
    weights: torch.Tensor
    idxs_src: torch.Tensor
    sbl: List[int]
    tbl: List[int]
    pos: List[int]
    af: List[int]
    delta: int


from .grappaND import *

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
