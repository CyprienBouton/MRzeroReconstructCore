import MRzeroCore as mr0
import torch
import numpy as np
import SimpleITK as sitk

from .RecoMRzero import RecoMRzero


class MRzeroReconstructLogic:
    @staticmethod
    def kspace_to_sitk(kspace: torch.Tensor):
        if sitk is None:
            raise RuntimeError("SimpleITK is required for image conversion.")
        """
        Convert Cartesian multi-coil k-space into a scalar volume
        suitable for visualization in 3D Slicer.

        The coil dimension is combined using root-sum-of-squares.
        The magnitude is log-compressed to visualize the large
        dynamic range of k-space.
        """

        # Combine coils using root-sum-of-squares
        kspace_magnitude = torch.sqrt(
            torch.sum(torch.abs(kspace) ** 2, dim=-1)
        )

        # Log compression
        kspace_display = torch.log1p(kspace_magnitude)

        # Normalize to [0, 1]
        kspace_display = kspace_display - kspace_display.min()

        max_value = kspace_display.max()

        if max_value > 0:
            kspace_display = kspace_display / max_value

        kspace_volume = kspace_display.permute(2, 1, 0).float()

        return MRzeroReconstructLogic.make_sitk_image(
            kspace_volume,
            spacing=(1.0, 1.0, 1.0),
            direction=(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ),
            center_volume=True
        )
        
    @staticmethod
    def center_origin(sitk_img):
        if sitk is None:
            raise RuntimeError("SimpleITK is required for image conversion.")
        spacing = np.array(sitk_img.GetSpacing())
        direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
        size = np.array(sitk_img.GetSize(), dtype=float)
        size_mm = size * spacing
        half_size = 0.5 * size_mm
        # Adjust origin so that (0,0,0) is at the physical center of the volume,
        # accounting for the fact that the origin in SimpleITK is the center of the first voxel
        origin = tuple(-direction @ (half_size - 0.5 * spacing))
        sitk_img.SetOrigin(origin)
        return sitk_img

    @staticmethod
    def make_sitk_image(
        volume: torch.Tensor,
        spacing: tuple = (1.0, 1.0, 1.0),
        origin: tuple = (0.0, 0.0, 0.0),
        direction: tuple = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        center_volume: bool = True):
        if sitk is None:
            raise RuntimeError("SimpleITK is required for image conversion.")
        """
        Convert a torch tensor to SimpleITK image with centered origin.
        Assumes volume shape is [Z, Y, X] or compatible.
        """
        data = volume.detach().cpu().numpy().astype(np.float32)
        # Convert to sitk image with spatial orientation and origin
        sitk_img = sitk.GetImageFromArray(data)

        sitk_img.SetSpacing(spacing)
        sitk_img.SetDirection(direction)
        if center_volume:
            sitk_img = MRzeroReconstructLogic.center_origin(sitk_img)
        else:
            sitk_img.SetOrigin(origin)

        return sitk_img

    def runReconstruction(
        self,
        signalPath,
        sequencePath,
        method,
        kernelSize=(4, 1, 5),
        reg=1e-4,
        tol=0.1,
        reg_cs=0.1,
        reference_scan='mode'
    ):
        if sequencePath.endswith('.seq'):
            if mr0 is None:
                raise RuntimeError("MRzeroCore is required to load .seq files.")
            seq0 = mr0.Sequence.import_file(
                sequencePath.replace('\\\\', '\\')
            )

        elif sequencePath.endswith('.pkl'):
            import pickle

            with open(sequencePath, 'rb') as f:
                seq0 = pickle.load(f)

        else:
            raise ValueError("Unsupported sequence file format")

        reco = RecoMRzero(seq0)

        # Load signal
        signal_np = np.load(signalPath)
        signal = torch.from_numpy(signal_np)

        # ============================================================
        # RAW CARTESIAN K-SPACE
        # ============================================================

        raw_kspace, kx_vals, ky_vals, kz_vals = reco.cartesian_kspace_grid(
            signal
        )

        # This is the original k-space before reconstruction processing
        raw_kspace_sitk = self.kspace_to_sitk(raw_kspace)

        # ============================================================
        # RECONSTRUCTION
        # ============================================================

        if method == "FFT":

            # Processed k-space used by the FFT
            reconstruction_kspace = reco.pad_kspace_to_center(
                raw_kspace,
                kx_vals,
                ky_vals,
                kz_vals
            )

            # Final reconstruction
            output = reco.fft_reconstruct(
                reconstruction_kspace
            )

        elif method == "GRAPPA":

            # Processed k-space before GRAPPA
            reconstruction_kspace = reco.pad_kspace_to_center(
                raw_kspace,
                kx_vals,
                ky_vals,
                kz_vals
            )

            # Extract ACS
            acs = reco.get_acs_from_kspace(
                reconstruction_kspace
            )

            # GRAPPA reconstruction
            output = reco.grappa_reconstruct(
                reconstruction_kspace,
                acs,
                kernel_size=kernelSize,
                reg=reg
            )

            # GRAPPA returns the reconstructed image.
            # Recover the corresponding k-space for visualization.
            if output.dim() >= 3:
                reconstruction_kspace = torch.fft.fftn(
                    output,
                    dim=tuple(range(output.dim() - 1))
                    if output.dim() == 4
                    else tuple(range(output.dim()))
                )

        elif method == "Zero-filled":

            output = reco.zero_filled_cs(
                raw_kspace,
                kx_vals,
                ky_vals,
                kz_vals,
                tolerance=tol,
                regularization=reg_cs,
                reference_scan=reference_scan
            )

            # Zero-filled CS returns an image-domain result.
            # Convert the complex reconstructed image back to k-space
            # so that the displayed k-space corresponds to the output.
            if output.dim() == 4:
                reconstruction_kspace = torch.fft.fftn(
                    output,
                    dim=(0, 1, 2)
                )
            else:
                reconstruction_kspace = torch.fft.fftn(
                    output,
                    dim=tuple(range(output.dim()))
                )

        else:
            raise NotImplementedError(
                f"Unknown reconstruction method: {method}"
            )

        # ============================================================
        # RECONSTRUCTED IMAGE
        # ============================================================

        if output.dim() == 2:
            magnitude = output.abs().unsqueeze(-1)

        elif output.dim() == 4:
            magnitude = output.abs().mean(dim=-1)

        else:
            magnitude = output.abs()
        
        if reconstruction_kspace.dim() == 3:
            reconstruction_kspace = reconstruction_kspace.unsqueeze(-1)

        sitk_volume = magnitude.permute(2, 1, 0)

        image_sitk = self.make_sitk_image(
            sitk_volume,
            spacing=reco.get_spacing()
        )

        # ============================================================
        # RECONSTRUCTION K-SPACE
        # ============================================================

        reconstruction_kspace_sitk = self.kspace_to_sitk(
            reconstruction_kspace
        )

        return (
            image_sitk,
            reconstruction_kspace_sitk,
            raw_kspace_sitk
        )