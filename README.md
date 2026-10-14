# MRzero Reconstruction Core

Standalone Python reconstruction core for MRzero MRI reconstruction, with support for GRAPPA, BART-related utilities, and integration with 3D Slicer.

## Overview

This project separates reconstruction algorithms from the 3D Slicer interface, making the core easier to maintain, test, and reuse independently.

### Main components

- `RecoMRzero.py`: Main reconstruction implementation.
- `ggrappa/`: GRAPPA-related reconstruction utilities.
- `bart_python/`: Python utilities for BART integration.
- `MRzeroReconstructLogic.py`: 3D Slicer module logic that connects the reconstruction core to Slicer.

## Project structure

```text
project/
├── README.md
├── pyproject.toml
├── MRzeroReconstructCoreLib/
│   ├── __init__.py
│   ├── RecoMRzero.py
│   ├── bart_python/
│   └── ggrappa/
└── MRzeroReconstructCore.py
```

Adapt this structure to match the actual repository layout.

## Requirements

- Python 3.9 or later
- NumPy
- PyTorch
- SimpleITK
- BART, if required by the selected reconstruction methods
- 3D Slicer, when using the Slicer interface

## Installation

### Install the Python package

From the project root, run:

```bash
python -m pip install -e .
```

This installs the package in editable mode, allowing source code changes to be used without reinstalling the package.

### Verify the installation

```bash
python -c "import MRzeroReconstructCore; print(MRzeroReconstructCore.__file__)"
```

### Configure BART

If the reconstruction requires the external BART toolbox, install and configure it separately according to its official instructions.

Ensure that the required executable, libraries, and environment variables are available to the Python environment running the reconstruction.

Installing this package does not automatically install the external BART toolbox.

## Usage

### Import the reconstruction core

```python
from MRzeroReconstructCore import MRzeroReconstructLogic
```

### Integration with 3D Slicer

The Slicer module logic can import the standalone reconstruction core:

Keep reconstruction algorithms and supporting utilities in `MRzeroReconstructCore`, and Slicer-specific operations in the Slicer module.

Ensure that the core package and its dependencies are accessible from Slicer's Python environment. Installing the package in your system Python environment does not necessarily make it available in Slicer.

## Development

### Editable installation

```bash
python -m pip install -e .
```

## Dependencies

The Python dependencies declared in `pyproject.toml` currently include:

- `numpy`
- `torch`
- `SimpleITK`

Update `pyproject.toml` whenever additional required Python dependencies are introduced.