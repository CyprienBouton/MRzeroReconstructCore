# MRzero Reconstruct Core

Python reconstruction core for **MRzero simulated MRI signals**. It grids the
simulated ADC samples of an [MRzero](https://github.com/MRsources/MRzero-Core)
sequence onto a Cartesian k-space and reconstructs them with a centred FFT,
GRAPPA or wavelet-regularized compressed sensing. Results are returned as SimpleITK images.

The package contains no 3D Slicer code, so it can be used and tested on its own.

| | |
|---|---|
| Distribution name | `mrzero-reconstruct-core` |
| Import package | `MRzeroReconstructCoreLib` |
| Sister package | `mrtwin-reconstruct-core` (Siemens twix raw data, same layout and API) |

## Used by

The 3D Slicer extension **MRzeroReconstruct** (`SlicerExtensions/MRzeroReconstruct`).
It imports `MRzeroReconstructCoreLib.MRzeroReconstructCore.MRzeroReconstructLogic`
and calls `runReconstruction(signal, sequence, method, kernelSize=..., reg=...,
tol=..., reg_cs=..., reference_scan=...)`. If the package is missing, the
extension installs it from this GitHub repository with `slicer.util.pip_install`.

## Reconstruction methods

| `method` | What it does | Needs |
|---|---|---|
| `"FFT"` | Zero-pads k-space so that k = 0 is centred, then applies a unitary centred inverse FFT. | — |
| `"NUFFT"` | Density-compensated adjoint NUFFT directly from exact sample coordinates, with explicit image dimensions and field of view. | torchkbnufft |
| `"GRAPPA"` | Calibrates a GRAPPA kernel on the fully sampled centre (ACS) and fills the missing lines (vendored `ggrappa`). The acceleration is detected from the sampling pattern unless `af` is given. | — (GPU used when available) |
| `"Zero-filled"` | For inversion-recovery sequences: zeroes the lines acquired with an irregular recovery time, then reconstructs with L1-wavelet compressed sensing. | — |

Method names are case-insensitive. FFT and GRAPPA return the root-sum-of-squares
(RSS) of the coil images. Zero-filled returns the magnitude of a coil-combined image.

## Installation

```bash
# from GitHub
python -m pip install "git+https://github.com/CyprienBouton/MRzeroReconstructCore.git"

# from a clone (editable, with test dependencies)
python -m pip install -e ".[test]"
```

Requirements: Python ≥ 3.9, `numpy`, `torch`, `scipy`, `tqdm`, `SimpleITK`,
`mrzerocore` (installed automatically). `mrzerocore` ≥ 1.0 needs Python ≥ 3.10;
on Python 3.9 (3D Slicer 5.8) pip selects `mrzerocore` 0.4.x, which provides
the same sequence API.

**In 3D Slicer**, install from the Slicer Python console
(`slicer.util.pip_install("git+https://github.com/CyprienBouton/MRzeroReconstructCore.git")`)
or let the extension do it. Installing into your system Python does not make
the package available inside Slicer.

### Zero-filled dependencies

Python 3.10 and later install [MRI-NUFFT](https://mind-inria.github.io/mri-nufft/)
with the CPU FINUFFT backend for multi-coil calibration. On Python 3.9
(3D Slicer 5.8), the calibration uses the equivalent Cartesian PyTorch FFT.
The iterative reconstruction uses PyTorch on the input tensor's device.

## Quick start

```python
import SimpleITK as sitk
from MRzeroReconstructCoreLib import MRzeroReconstructLogic

logic = MRzeroReconstructLogic()
image, kspace, raw_kspace = logic.runReconstruction(
    "signal.npy",          # simulated signal [n_samples, ncoils]
    "sequence.seq",        # or a pickled MRzeroCore.Sequence (.pkl)
    "GRAPPA",
    kernelSize=(4, 1, 5),  # (ky, kz, kx)
)
sitk.WriteImage(image, "image.nii.gz")
```

Command line: `python examples/reconstruct_mrzero.py signal_*.npy --sequence seq.seq --method FFT -o out/`.

## API

### `MRzeroReconstructLogic` (`MRzeroReconstructCoreLib.MRzeroReconstructCore`)

`runReconstruction(signalPath, sequencePath, method="FFT", kernelSize=(4, 1, 5), reg=1e-4, tol=0.1, reg_cs=0.1, reference_scan="mode", signalPathRef=None, gaussian_noise_factor=0.0, doRemoveNavigator=True, af=None)`

| Argument | Meaning |
|---|---|
| `signalPath` | `.npy` simulated signal `[n_samples, ncoils]` |
| `sequencePath` | `.seq` (Pulseq) or `.pkl` (pickled `MRzeroCore.Sequence`; only load trusted files) |
| `method` | `"FFT"`, `"GRAPPA"` or `"Zero-filled"` |
| `kernelSize`, `reg`, `af` | GRAPPA kernel `(ky, kz, kx)`, Tikhonov regularisation, acceleration `(af_y, af_z)` (`None` = detected) |
| `tol`, `reg_cs`, `reference_scan` | Zero-filled: relative tolerance on the recovery time, L1-wavelet weight, reference recovery time (`"mode"` or `"min"`) |
| `gaussian_noise_factor`, `signalPathRef` | Adds complex Gaussian noise with std = factor × max of the reference signal (the signal itself by default) |
| `doRemoveNavigator` | Discard navigator blocks of inversion-recovery sequences |

Returns `(image_sitk, reconstruction_kspace_sitk, raw_kspace_sitk)`:
the RSS magnitude image (spacing in mm from the k-space step, centred on the
origin), the k-space that produced it and the gridded raw k-space. The two
k-space volumes are log-compressed RSS magnitudes normalised to [0, 1], for display.

`runMultipleReconstructions(signalPaths, sequencePath, method="FFT", **kwargs)`
runs several signals and returns a list of `{"filename", "result", "error"}`.

Static helpers: `kspace_to_sitk`, `make_sitk_image`, `center_origin`,
`add_gaussian_noise`. `load_sequence(path)` is also exported.

### `RecoMRzero` (lower level)

| Method | Purpose |
|---|---|
| `remove_navigator()` | Disable navigator ADCs / pulses (sequence modified in place) |
| `cartesian_kspace_grid(signal)` | Grid the signal → raw k-space `[nx, ny, nz, ncoils]` |
| `center_kspace(kspace)` | Zero-pad so that k = 0 is at `n // 2` (no sample discarded) |
| `fft_reconstruct(kspace)` | Unitary centred inverse FFT → coil images |
| `get_acs_from_kspace(kspace, center)` | Largest fully sampled block around the centre |
| `grappa_reconstruct(kspace, acs, af, kernel_size, reg)` | GRAPPA → filled k-space |
| `zero_filled_cs(raw_kspace, tolerance, regularization, reference_scan)` | L1-wavelet compressed sensing → complex image |
| `get_spacing(shape)` | Voxel size in mm for a grid of `shape` |

### `reco_utils`

Centred FFTs, centring by zero-padding, ACS / acceleration detection, GRAPPA,
POCS partial Fourier and SimpleITK conversion.

## Conventions and methods

* **Arrays.** k-space is a complex tensor `[nx, ny, nz, ncoils]`
  (readout, phase, partition, coil); unacquired samples are exactly zero.
* **Gridding.** `seq.get_kspace()` gives the k-space position [1/m] of every
  ADC sample. Positions are rounded to 0.05 m⁻¹; the k-space step is measured
  next to k = 0 (which must be fully sampled), and every gap between
  consecutive positions is rounded to a multiple of the step, so undersampling
  gaps stay on the grid. The voxel size is `1 / (Δk · n)`. A single partition
  gets a nominal 5 mm slice.
* **Centring.** k = 0 is moved to index `n // 2` by zero-padding (the
  `fftshift` convention). Asymmetric / partial-Fourier axes
  are padded to an even size; nothing is cropped.
* **GRAPPA.** Each missing sample is a linear combination of the acquired
  neighbours of all coils, `s_j(k) = Σ_c Σ_d w_{j,c,d} s_c(k + d)`. The
  weights are fitted on the ACS by Tikhonov-regularised least squares,
  `W = (SᴴS + λ²σ_max I)⁻¹ SᴴT` with `λ = reg`.
* **Compressed sensing.** FISTA solves
  `argmin_x ½‖P F S x − y‖² + λ‖Ψx‖₁` using centered PyTorch FFTs and
  orthonormal Haar wavelets. The mask P selects acquired samples with eligible
  recovery times. Measurements y are scaled by the largest retained magnitude.
  Coil sensitivities S are estimated from normalized low-frequency coil
  images using a central calibration window (default `6:6:2`).

## Repository structure

```text
MRzeroReconstructCore/
├── README.md
├── pyproject.toml
├── .gitignore
├── MRzeroReconstructCoreLib/
│   ├── __init__.py               # public API, __version__
│   ├── MRzeroReconstructCore.py  # MRzeroReconstructLogic (used by Slicer)
│   ├── RecoMRzero.py             # gridding + reconstruction back-ends
│   ├── reco_utils.py             # Cartesian reconstruction utilities
│   ├── compressed_sensing.py    # wavelet-regularized SENSE
│   └── ggrappa/                  # vendored GRAPPA (BSD-3-Clause, see LICENSE)
├── examples/
│   └── reconstruct_mrzero.py
└── tests/                        # synthetic-data tests, no data files needed
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The tests use a synthetic multi-coil phantom and a fake sequence.

## Licences

* **This package:** no licence has been chosen yet. Add a `LICENSE` file
  before publishing. Note that `mrzerocore` is AGPL-3.0, which matters if you
  distribute this package together with it.
* **`ggrappa/`** — vendored from [gGRAPPA](https://github.com/mbertrait/ggrappa)
  (Maxime Bertrait and gGRAPPA collaborators) through the fork
  [CyprienBouton/ggrappa](https://github.com/CyprienBouton/ggrappa), commit
  `39cfab0`. BSD 3-Clause, `ggrappa/LICENSE` (upstream's `pyproject.toml`
  says MIT but its `LICENSE` file is BSD-3-Clause). Local change: the kernel
  sliding range in `application.py` (`- sbl + 1` instead of `- sbl - 1`), which
  left the last k-space lines/partitions unfilled.
* Runtime dependencies: `mrzerocore` (AGPL-3.0), PyTorch (BSD-style),
  NumPy/SciPy (BSD-3-Clause), SimpleITK (Apache-2.0), tqdm (MIT/MPL-2.0).

## Known limitations

* GRAPPA acceleration detection assumes a regular Cartesian pattern (no CAIPIRINHA shift).
* `remove_navigator()` relies on a specific sequence layout (inversion pulses
  in pairs two repetitions apart); check it on new sequences.
* The `"Zero-filled"` method is only meaningful for inversion-recovery sequences
  with enough central k-space for coil calibration.

The NUFFT method uses `torchkbnufft` for compatibility with Slicer Python 3.9.
It weights samples for their sampling density before applying the adjoint
nonuniform Fourier transform and combining coil magnitudes. It bypasses
Cartesian gridding for radial and spiral acquisitions.

Set `nufft_shape=(nx, ny, nz)` and `nufft_fov=(fx, fy, fz)` in metres to
match the simulation. Defaults are `(128, 128, 1)` and
`(0.2, 0.2, 0.005)`. Slicer's NUFFT geometry controls accept field of view
in millimetres. A singleton axis requires zero trajectory coordinates.
For NUFFT the third returned display volume contains the reconstructed
Cartesian spectrum, since nonuniform raw samples have no Cartesian grid.
