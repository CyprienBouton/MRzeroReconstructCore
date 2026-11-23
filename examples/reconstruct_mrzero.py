"""Reconstruct MRzero simulated signals and save the results as NIfTI.

Example::

    python examples/reconstruct_mrzero.py signal_*.npy --sequence seq.seq --method GRAPPA -o out/

Writes ``<name>_<method>_image.nii.gz``, ``<name>_<method>_kspace.nii.gz``
and ``<name>_raw_kspace.nii.gz`` for every input signal.
"""

import argparse
import os

import SimpleITK as sitk

from MRzeroReconstructCoreLib import METHODS, MRzeroReconstructLogic


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("signals", nargs="+", help="simulated signal(s), .npy")
    parser.add_argument("--sequence", required=True, help="MRzero sequence, .seq or .pkl")
    parser.add_argument("--method", default="FFT", choices=METHODS)
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian noise level relative to the signal maximum")
    parser.add_argument("-o", "--output", default=".", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = MRzeroReconstructLogic().runMultipleReconstructions(
        args.signals, args.sequence, method=args.method,
        gaussian_noise_factor=args.noise)

    for entry in results:
        name = os.path.splitext(os.path.basename(entry["filename"]))[0]
        if entry["error"] is not None:
            print(f"[ERROR] {name}: {entry['error']}")
            continue
        image, kspace, raw_kspace = entry["result"]
        prefix = os.path.join(args.output, name)
        sitk.WriteImage(image, f"{prefix}_{args.method}_image.nii.gz")
        sitk.WriteImage(kspace, f"{prefix}_{args.method}_kspace.nii.gz")
        sitk.WriteImage(raw_kspace, f"{prefix}_raw_kspace.nii.gz")
        print(f"[OK] {name} -> {prefix}_*.nii.gz")


if __name__ == "__main__":
    main()
