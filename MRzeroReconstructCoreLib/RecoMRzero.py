import torch
import numpy as np
import torch.nn.functional as F

from . import bart_python  # BART python binding
from . import ggrappa


def extract_indices_from_kspace_coords(kspace_coords: torch.Tensor, dk: float) -> torch.Tensor:
    """
    Compute discrete k-space indices for each coordinate in the input array.
    This avoids cumulative floating-point error by rounding each gap individually.

    Args:
        kspace_coords (torch.Tensor): 1D tensor of k-space coordinates (e.g., phase-encode positions).
        dk (float): Expected k-space step size along this axis.

    Returns:
        torch.Tensor: 1D tensor of integer indices, same shape as kspace_coords.
    """
    device = kspace_coords.device

    # Get sorted unique coordinates and inverse mapping
    unique_coords, inverse = torch.unique(kspace_coords, sorted=True, return_inverse=True)

    # Compute relative step sizes between consecutive coordinates,
    # normalized by dk and rounded to the nearest integer
    gaps = torch.round(torch.diff(unique_coords) / dk).to(torch.long)

    # Insert a zero at the beginning so length matches unique_coords
    gaps = torch.cat([torch.zeros(1, dtype=torch.long, device=device), gaps])

    # Cumulative sum → integer indices for unique coords
    unique_indices = torch.cumsum(gaps, dim=0)

    # Map each original coordinate back to its index
    return unique_indices[inverse]


class RecoMRzero:
    def __init__(self, seq0, dx=None, dy=None, dz=None):
        self.seq0 = seq0
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def cartesian_kspace_grid(self, signal, precision=5e-2,
                              dtype=torch.complex64, device='cpu'):
        """
        Build a Cartesian k-space grid from non-uniform k-space coordinates and signal.

        Returns:
            grid: tensor [nx, ny, nz, ncoils]
            kx_vals, ky_vals, kz_vals: 1D tensors of grid coordinates
        """
        kspace_coords = self.seq0.get_kspace()
        kx, ky, kz = kspace_coords[:, :3].T

        # Round coordinates
        kx_rounded = ((kx / precision).round() * precision)
        ky_rounded = ((ky / precision).round() * precision)
        kz_rounded = ((kz / precision).round() * precision)

        # Get sorted unique coordinates
        kx_rounded_unique = kx_rounded.unique().sort().values
        ky_rounded_unique = ky_rounded.unique().sort().values
        kz_rounded_unique = kz_rounded.unique().sort().values

        # Determine step sizes if not provided assume fully sampled in center
        if self.dx is None:
            self.dx = kx_rounded_unique.diff()[kx_rounded_unique.abs().argmin()]
        if self.dy is None:
            self.dy = ky_rounded_unique.diff()[ky_rounded_unique.abs().argmin()]
        if self.dz is None:
            if len(kz_rounded_unique) > 1:
                self.dz = kz_rounded_unique.diff()[kz_rounded_unique.abs().argmin()]
            else:
                self.dz = 1/(5e-3)  # Assume 5mm spacing if only one slice

        # Compute nearest grid indices
        self.ix = extract_indices_from_kspace_coords(kx_rounded, self.dx)
        self.iy = extract_indices_from_kspace_coords(ky_rounded, self.dy)
        self.iz = extract_indices_from_kspace_coords(kz_rounded, self.dz)

        # Define grid ranges
        self.nx, self.ny, self.nz = self.ix.max()+1, self.iy.max()+1, self.iz.max()+1
        self.ncoils = signal.shape[1]

        grid = torch.zeros((self.nx, self.ny, self.nz, self.ncoils), dtype=dtype, device=device)
        # Fill the grid safely (accumulate for repeated coordinates)
        grid.index_put_((self.ix, self.iy, self.iz), signal.to(dtype), accumulate=True)

        return grid, kx_rounded_unique, ky_rounded_unique, kz_rounded_unique

    def get_acs_from_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Extract the largest fully sampled ACS region from 3D k-space.

        Args:
            kspace (torch.Tensor): Full k-space data [nx, ny, nz, ncoils]

        Returns:
            torch.Tensor: ACS region [n_acsx, n_acsy, n_acsz, ncoils]
        """
        mask = kspace.abs().sum(dim=-1) > 0  # [nx, ny, nz]

        full_x = mask.any(dim=(1, 2))
        full_y = mask.any(dim=(0, 2))
        full_z = mask.any(dim=(0, 1))

        def contiguous_true_indices(bool_tensor):
            indices = torch.where(bool_tensor)[0]
            if len(indices) == 0:
                return None, None
            diffs = torch.diff(indices)
            breaks = torch.where(diffs > 1)[0]
            starts = torch.cat((indices[0:1], indices[breaks + 1]))
            ends = torch.cat((indices[breaks], indices[-1:]))
            lengths = ends - starts + 1
            max_idx = torch.argmax(lengths)
            return starts[max_idx].item(), ends[max_idx].item() + 1

        x_start, x_end = contiguous_true_indices(full_x)
        y_start, y_end = contiguous_true_indices(full_y)
        z_start, z_end = contiguous_true_indices(full_z)

        if None in (x_start, x_end, y_start, y_end, z_start, z_end):
            raise ValueError("No fully sampled ACS region found in k-space.")

        acs = kspace[x_start:x_end, y_start:y_end, z_start:z_end, :]

        return acs
    
    def _bart_python_reconstruct(self, kspace: torch.Tensor,
                                bart_cmd: str) -> torch.Tensor:
        """
        Run a BART command on k-space data, handling tensor formatting and device.

        Args:
            kspace: torch.Tensor [nx, ny, nz, ncoils]
            bart_cmd: str, BART command (e.g. 'fft -i -d 3')

        Returns:
            torch.Tensor: Output tensor from BART command
        """
        # bart_input = kspace.permute(3, 1, 2, 0).contiguous()

        output_np = bart_python.bart(1, bart_cmd, kspace.cpu().numpy())

        # Convert back to torch and restore device
        output_tensor = torch.from_numpy(output_np).to(kspace.device)

        return output_tensor

    def grappa_reconstruct(self, kspace: torch.Tensor, acs: torch.Tensor,
                           af=[2,1], kernel_size=(4,4,5), reg=1e-4,
                           batch_size=1, cuda=True, quiet=True) -> torch.Tensor:
        """
        Perform GRAPPA reconstruction using the ggrappa implementation.

        Args:
            kspace: torch.Tensor [nx, ny, nz, ncoils]
            acs: torch.Tensor ACS region [n_acsx, n_acsy, n_acsz, ncoils]
            af: list or tuple, acceleration factors [afy, afz]
            kernel_size: GRAPPA kernel size
            reg: regularization lambda
            batch_size: batch size for kernel application
            cuda: use GPU
            quiet: suppress output

        Returns:
            torch.Tensor: Reconstructed kspace [nx, ny, nz, ncoils]
        """
        # Reorder from [nx, ny, nz, ncoils] to [ncoils, ky, kz, kx] for ggrappa
        kspace_bart = kspace.permute(3, 1, 2, 0).contiguous()
        acs_bart = acs.permute(3, 1, 2, 0).contiguous()

        # Run GRAPPA reconstruction
        reco_kspace, _ = ggrappa.GRAPPA_Recon(
            sig=kspace_bart,
            acs=acs_bart,
            af=af,
            kernel_size=kernel_size,
            lambda_=reg,
            batch_size=batch_size,
            cuda=cuda,
            quiet=quiet
        )

        # Convert back to [nx, ny, nz, ncoils]
        reco_kspace = reco_kspace.permute(3, 1, 2, 0).contiguous()

        return self.fft_reconstruct(reco_kspace)
    
    def fft_reconstruct(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Perform FFT reconstruction on the kspace data using BART.

        Args:
            kspace: torch.Tensor [nx, ny, nz, ncoils]
            bart_slc_axis: int, axis used for slice calibration (default=0)

        Returns:
            torch.Tensor: Reconstructed image data
        """
        bitmask = 7  # dims 0,1,2 -> 1<<0 + 1<<1 + 1<<2 = 7
        bart_cmd = f'fft -i -u {bitmask}'
        return self._bart_python_reconstruct(kspace, bart_cmd)

    def pad_kspace_to_center(self, kspace, kx_vals, ky_vals, kz_vals):
        """
        Shift the Cartesian k-space data so that the sample closest to
        physical k=0 is located at the center of each array dimension.

        No wrapping is performed. Samples shifted outside the array are discarded.
        """

        def zero_index(k_vals):
            return torch.argmin(torch.abs(k_vals)).item()

        def shift_axis(src, shift, axis):
            dst = torch.zeros_like(src)

            if shift >= 0:
                src_start = 0
                src_end = src.shape[axis] - shift
                dst_start = shift
                dst_end = src.shape[axis]
            else:
                src_start = -shift
                src_end = src.shape[axis]
                dst_start = 0
                dst_end = src.shape[axis] + shift

            src_slices = [slice(None)] * src.ndim
            dst_slices = [slice(None)] * src.ndim

            src_slices[axis] = slice(src_start, src_end)
            dst_slices[axis] = slice(dst_start, dst_end)

            dst[tuple(dst_slices)] = src[tuple(src_slices)]

            return dst

        nx, ny, nz, nc = kspace.shape

        # Location of physical k = 0
        zero_x = zero_index(kx_vals)
        zero_y = zero_index(ky_vals)
        zero_z = zero_index(kz_vals)

        # Desired array center
        center_x = nx // 2
        center_y = ny // 2
        center_z = nz // 2

        # Required shifts
        shift_x = center_x - zero_x
        shift_y = center_y - zero_y
        shift_z = center_z - zero_z

        # Apply shifts without wrapping
        centered = shift_axis(kspace, shift_x, axis=0)
        centered = shift_axis(centered, shift_y, axis=1)
        centered = shift_axis(centered, shift_z, axis=2)

        return centered
    
    def get_spacing(self):
        # Correct voxel spacing in mm
        spacing_x = 1.0 / (self.dx * self.nx)
        spacing_y = 1.0 / (self.dy * self.ny)
        spacing_z = 1.0 / (self.dz * self.nz)

        spacing = (spacing_x.item()*1e3, spacing_y.item()*1e3, spacing_z.item()*1e3)
        return spacing
    
    def zero_filled_cs(self, kspace: torch.Tensor, kx_vals, ky_vals, kz_vals, 
                       tolerance=0.15, regularization=0.1, reference_scan='mode') -> torch.Tensor:
        inter_scan_matrix = self.get_inter_scan_duration()
        matrix_flat = inter_scan_matrix.flatten()
        matrix_flat = torch.round(matrix_flat * 10) / 10
        matrix_flat = matrix_flat[~torch.isnan(matrix_flat)]
        if reference_scan=='mode':
            regular_inter_scan, _ = torch.mode(matrix_flat)
        elif reference_scan=='min':
            regular_inter_scan = matrix_flat.min()
        else:
            raise ValueError("reference_scan must be 'mode' or 'min'")
        # --------------------- ADDED BLOCK ---------------------
        # Zero-out k-space samples where inter-scan duration > 1.2
        bad_mask = (abs(inter_scan_matrix-regular_inter_scan) > tolerance*regular_inter_scan)            # shape: (ny, nz)

        # reshape mask to broadcast over (nx, ny, nz, ncoils)
        bad_mask4d = bad_mask.unsqueeze(0).unsqueeze(-1)   # (1, ny, nz, 1)
        sensitivities = bart_python.bart(1, "ecalib -m1 -k 6:6:2", kspace.cpu().numpy())
        while sensitivities.ndim < kspace.ndim:
            sensitivities = sensitivities[:, :, :, np.newaxis]
        sensitivities = self.pad_kspace_to_center(torch.from_numpy(sensitivities), kx_vals, ky_vals, kz_vals)
        kspace_filled = kspace.masked_fill(bad_mask4d, 0)
        # -------------------------------------------------------
        kspace = self.pad_kspace_to_center(kspace, kx_vals, ky_vals, kz_vals)
        kspace_filled = self.pad_kspace_to_center(kspace_filled, kx_vals, ky_vals, kz_vals)
        kspace_np = kspace_filled.cpu().numpy()
        sens_np = sensitivities.cpu().numpy()
        kspace_np /= np.abs(kspace_np).max()

        pics_cmd = 'pics'
        if torch.cuda.is_available():
            pics_cmd += ' -g'
        pics_cmd += f' -R W:7:0:{regularization} -S -n'

        output_np = bart_python.bart(1, pics_cmd, kspace_np, sens_np)
        # Handle 2D images
        if output_np.ndim == 2:
            output_np = output_np[:, :, np.newaxis]  
            
        output = torch.from_numpy(output_np).to(kspace.device).abs()
        return output
    
    def get_inter_scan_duration(self, eps=0.2, dtype=torch.float, device='cpu'):
        inter_scan_matrix = torch.full((self.ny, self.nz), torch.nan, dtype=dtype, device=device)
        times_after = torch.cumsum(torch.tensor([r.event_time.sum() for r in self.seq0]), dim=0)
        times_before = torch.tensor([ 0. ] + list(times_after[:-1]))
        IR_idx = [i for i, r in enumerate(self.seq0) if abs( r.pulse.angle - torch.pi ) < eps]
        times_IR = times_before[IR_idx]
        inter_scan_durations = torch.diff(times_IR, prepend=torch.tensor([torch.nan]) )
        times_IR_next = torch.cat( [times_IR[1:], torch.tensor([torch.inf])] )
        for time_IR, time_IR_next, duration in zip(times_IR, times_IR_next, inter_scan_durations):
            current_shot = (times_before>time_IR)*(times_before<time_IR_next)
            mask_signal = torch.cat([
                torch.full((int(r.adc_usage.sum()),), is_shot, dtype=torch.int64)
                for is_shot, r in zip(current_shot, self.seq0)
            ]).to(bool)
            inter_scan_matrix[self.iy[mask_signal], self.iz[mask_signal]] = duration
        return inter_scan_matrix